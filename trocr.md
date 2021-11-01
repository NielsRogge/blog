## Introduction

Transformer-based models have overtaken a lot of fields in AI already. It started with [NLP](https://arxiv.org/abs/1706.03762) in 2017, but they quickly started to match or outperform state-of-the-art algorithms in other domains (including [computer vision](https://arxiv.org/abs/2010.11929), [speech recognition](https://arxiv.org/abs/1904.05862), [computational biology](https://www.nature.com/articles/s41586-021-03819-2)). As it turns out, Transformers can also be applied to domains one would maybe not expect, such as table question answering (as shown by [TAPAS](https://arxiv.org/abs/2004.02349)) and intelligent document processing (as shown by [LayoutLM](https://arxiv.org/abs/1912.13318), [LayoutLMv2](https://arxiv.org/abs/2012.14740)). One typically needs to come up with a creative way to provide the data to a Transformer, as well as add some useful inductive biases (such as additional embedding layers) in order for these models to work well on a new domain.

Microsoft has been pretty busy making Transformers work for intelligent document processing. They introduced LayoutLM in 2019, a Transformer encoder (inspired by BERT) which can be used for document understanding tasks, such as classifying scanned documents, or extracting information from them. In 2020, they improved LayoutLM by integrating visual features during pre-training, with a new model called LayoutLMv2. Both LayoutLM and LayoutLMv2 rely on an external optical character recognition (OCR) algorithm, such as Google's Tesseract or Microsoft's Azure Cognitive Services. While existing OCR engines are definitely useful to extract text from scanned documents, there is still large room for improvement, as existing models still struggle on more challenging documents, and especially handwritten text.

This year, Microsoft came up with a new algorithm called [TrOCR](https://arxiv.org/abs/2109.10282), which applies Transformers to the task of optical-character recognition in an end-to-end manner. Surprisingly, TrOCR is relatively simple: it uses an encoder-decoder architecture (similar to the original Transformer, or models such as BART and T5). However, in contrast to the aforementioned models, its encoder and decoder are applied to different modalities: it consists of an image Transformer as encoder and a text Transformer as decoder. The architecture is shown in the figure below (taken from the original [paper](https://arxiv.org/abs/2109.10282)).

![snippet](assets/35_trocr/trocr_architecture.png)

Note that TrOCR works on single-line text images: one requires a text detector first, before providing every single-line image to the model.

Images are presented to the model in a similar manner as the original Vision Transformer, namely as a sequence of fixed-size, non-overlapping patches (typically of resolution 16x16 or 32x32). After linearly embedding those patches, one also adds absolute (learnable) position embeddings, before providing those to the image Transformer encoder. The Image Transformer encoder then updates the hidden states of each of the patches using a mechanism called multi-head attention. The output of the encoder is a tensor of shape (batch_size, sequence_length, hidden_size), containing the final hidden states of each of the patches of the original image. The TrOCR authors resize every image to a resolution of 384x384, and used a patch resolution of 16x16. Hence, each image consists of (384//16)^2 = 576 patches. You can view these patches as "tokens", for which the model learns embeddings (similar to the `input_ids` of BERT). One typically also adds a special [CLS] token at the beginning of the sequence, whose final hidden state can be viewed as a representation of an entire image. Hence, the sequence length of the image encoder of TrOCR equals 577. The final hidden states of the encoder will be of shape `(batch_size, sequence_length, hidden_size)`. If we only forward a single image through the encoder, the `batch_size` equals 1. If we assume that we use a based-size encoder, then the `hidden_size` equals 768 (similar to BERT-base). Hence, the shape of the `encoder_hidden_states` is `(1, 577, 768)`. Next, these are provided to the text decoder. 

The text decoder is trained to autoregressively generate the text that occurs in the image. It will do so by cross-attending to the encoder hidden states. Text Transformers typically operate on subword tokens (meaning that a given word like "hello" may be tokenized into multiple tokens such as "hel" and "lo"). During training, the decoder is trained using a technique called "teacher forcing", meaning that one gives the correct previous token for the prediction of each next token. Imagine that the decoder needs to generate the text "hello world" for a given image. As the decoder operates on subwords, it would need to output ["hel", "lo"], hence `labels` equals . What we do during training, is that we first create the `decoder_input_ids`, which are provided as input to the decoder. The `decoder_input_ids` are equal to the `labels`, but shifted one position to the right and prepended with the ID of a special token called the `decoder_start_token_id`. This becomes more clear in a figure:


         ...            hel      lo   world     ...   => labels
          ↓               ↓       ↓      ↓        ↓
┌──────────────────----------------------------------
│                      DECODER                       │
└──────────────────----------------------------------
          ↓               ↓       ↓      ↓       ↓
 decoder_start_token     ...     hel    lo     world   => decoder_input_ids

Note that the decoder is trained in parallel, meaning that we send all `decoder_input_ids` through the decoder in a single forward pass. It's only at inference time that the decoder will autoregressively (i.e. sequentially, from left to right) generate the tokens, by feeding its prediction at time step i as input to time step i+1. During training, we don't do this, we simply provide the ground truth `decoder_input_ids` as input to the decoder, take its prediction from the last `decoder_hidden_states`, and compare this with the ground truth `labels` in order to calculate the cross-entropy loss.

The authors initialize the weights of the image encoder with those of [BEiT](https://arxiv.org/abs/2106.08254) (a strong self-supervised Vision Transformer) and the weights of the text decoder with those of [RoBERTa](https://arxiv.org/abs/1907.11692) (a strong self-supervised Text Transformer). The weights of the cross-attention layers of the decoder are randomly initialized. Next, the authors further pre-train this encoder-decoder model on millions of (partially synthetic) image-text pairs. The authors refer to this pre-training as "stage 1". Next, one can fine-tune TrOCR on a custom dataset, by simply further training the encoder-decoder model. Hence, fine-tuning is identical to pre-training for TrOCR (in contrast to models like BERT).

## Using TrOCR in HuggingFace Transformers
As part of the TrOCR release, we are adding a new generic class called `VisionEncoderDecoderModel`, which allows to combine any image Transformer encoder (such as ViT, DeiT, BEiT) with any text Transformer as decoder (such as BERT, RoBERTa, GPT-2). Let's take a look at how such a model can be used to perform inference on a new image.

After installing Transformers:

```bash
!pip install -q transformers
```

One can load a TrOCR model as follows:

```python
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten)
```

Note that this is just one of the checkpoints Microsoft released. You can find all of them on the [hub](https://huggingface.co/models?other=trocr). This particular checkpoint has been fine-tuned on the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), a large collection of annotated images of handwritten text.

In order to prepare data for the model, one can use `TrOCRProcessor`, which is just a wrapper around a `ViTFeatureExtractor` and a `RobertaTokenizer`. By default, calling the processor is equivalent to calling the feature extractor, as can be seen by typing: 

```python
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
print(processor.current_processor)
```
which prints:

```bash

```

Let's take an image of a single-line handwritten text, and prepare it for the model. Here I'm loading an image from the IAM Database. We load the image using Pillow, and feed that to the processor. We also specify `return_tensors="pt"` ("pt" is short for PyTorch), as the model implementation is in PyTorch.

```python
from PIL import Image
import requests

url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(image, return_tensors="pt").pixel_values
``` 

The processor (actually, the feature extractor) will apply some very basic image processing to the image, namely it will resize the image to resolution 384x384 and it will normalize the channels using a mean and standard deviation of (0.5, 0.5, 0.5). The `pixel_values` are of shape `(batch_size, num_channels, height, width)`, which in this case will be `(1, 3, 384, 384)`.

Let's now autoregressively generate text for this image! It's as simple as calling the generate method:

```python
generated_ids = model.generate(pixel_values)
```

By default, the `generate` method uses greedy decoding, meaning that it takes the token with the highest probability at each time step, and feed that as input to the model for the next time step. There are more complicated and fancy decoding methods available, such as beam search and top-k sampling. For a good overview, I refer to this excellent [blog post](https://huggingface.co/blog/how-to-generate) by our very own Patrick.

The model just outputs token ids, so we still need to turn those back into text. We can do so using the `decode` method of the processor (actually, the tokenizer), as follows:

```python
generated_text = processor.decode(generated_ids, skip_special_tokens=True)
print(generated_text)
```
which prints:

```bash

```

So that's for the inference part! I've just explained everything you need to understand how I made the Gradio demo that you can play around with [here](https://huggingface.co/spaces/nielsr/TrOCR-handwritten). 
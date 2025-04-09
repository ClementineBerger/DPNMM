---
title: ""
layout: single
permalink: /
author_profile: false
classes: wide
header:
  overlay_color: "#cc3131" #"#000"
  overlay_filter: "0.3"
excerpt: #"title"
---

You will find audio examples [here](./audio.md) and additional results [here](./results.md).

# Abstract

People often listen to music in noisy environments, seeking to isolate themselves from ambient sounds. Indeed, a music signal can mask some of the noise's frequency components due to the effect of simultaneous masking. In this article, we propose a neural network based on a psychoacoustic masking model, designed to enhance the music's ability to mask ambient noise by reshaping its spectral envelope with predicted filter frequency responses. The model is trained with a perceptual loss function that balances two constraints: effectively masking the noise while preserving the original music mix and the user's chosen listening level. We evaluate our approach on simulated data replicating a user's experience of listening to music with headphones in a noisy environment. The results, based on defined objective metrics, demonstrate that our system improves the state of the art.
{: .text-justify}

**Index Terms** - Ambient noise masking, deep filtering, psychoacoustics

## Erratum
A few typos crept into the text of the paper (but fortunately not into the code or the results):
- In the NMR metric, the absolute values are in fact simple brackets.
- In the architecture figure all the dimensions of size 64 are in fact of size 32.

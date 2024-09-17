---
title: ""
layout: single
permalink: /results
author_profile: false
header:
  overlay_color: "#cc3131" #"#000"
  overlay_filter: "0.3"
excerpt: #"title"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

On this page you can find the results presented in the article as well as additional ones that explore the performance of the system according to music genre, type of headphones, environment etc. 
{: .text-justify}

# Metrics

Two objective metrics are considered to perform the evaluation of the models. We compute a mean Noise-to-Mask Ratio (NMR) per audio sample of the test set, only selecting the Bark bands where the initial music masking threshold is below the noise level :
{: .text-justify}

$$\text{NMR} = \frac{1}{M} \sum_{n, \nu} (1-m_\nu(n)) | P_{dB}^{noise}(n,\nu) - \hat{T}_{dB}(n,\nu) |,$$

where $M = \sum_{n, \nu} (1-m_\nu(n))$ with $m_\nu$ a mask such that $m_\nu(n) = 0$ if the initial threshold is below the noise, and $m_\nu(n) = 1$ otherwise. The obtained NMR is compared to the initial NMR with the unprocessed music to evaluate how much the system can improve the masking effect on the bands where it is required. However, the system may as well induce power variations in the other bands. To evaluate this effect we also compute a mean Global Level Difference (GLD) : 
{: .text-justify}

$$\text{GLD} = \frac{1}{N} \sum | \hat{\mathcal{P}}_{dBA}^{music}(n) - \mathcal{P}_{dBA}^{music}(n) | .$$

Both metrics are computed by frequency ranges: broadband, first third of Bark bands (low), second third (medium), and last third (high). 
{: .text-justify}
# Results

## General results (presented in the article)

<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title></title>
    <style>
        /* Conteneur pour centrer les images */
        .image-container {
            text-align: center; /* Centre le contenu à l'intérieur du conteneur */
        }
        /* Style des images */
        .image-container img {
            display: block; /* Affiche les images comme des éléments de bloc */
            margin: 20px auto; /* Centre les images horizontalement avec une marge automatique */
            width: 80%; /* Définit la largeur des images à 80% du conteneur parent */
            max-width: 800px; /* Optionnel : limite la largeur maximale des images */
            height: auto; /* Maintient le ratio d'aspect des images */
        }
    </style>
</head>
<body>
    <div class="image-container">
        <!-- Les images à afficher côte à côte -->
        <img src="figures/nmr-1.png" alt="Image 1">
        <img src="figures/gld-1.png" alt="Image 2">
    </div>
</body>
</html>

## Earbuds impact

The noises on the test are filtered with the frequency responses of 3 models of earbuds to reproduce their respective passive attenuations : 
- Bose headphones QuietComfort
- Sony earbuds WF-1000XM4 with sound isolating sleeves
- Apple Airpods with smooth tips
{: .text-justify}

![image-center](figures/earbuds_fr.png){: .align-center}

The Bose and Sony headphones act as low-pass filters while the Airpods have a much smoother effect.
{: .text-justify}

Per model of headphones the obtained results are :

![image-center](figures/earbuds_nmr.png){: .align-center}
![image-center](figures/earbuds_gld.png){: .align-center}





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

On this page you can find the results presented in the article as well as additional ones that explore the performance of the system according to music genre, type of headphones, environment etc. The models evaluated are:
- Baseline model by Estreder et al.
- Proposed model DPNMM without any power constraint
- DPNMM with $\Delta \mathcal{P}_{max} = $ 2, 1, 0.5 dBA
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

In terms of NMR, all three versions of PDNMM outperform Estreder's model on the broadband metric statistically significantly, except DPNMM with $\Delta \mathcal{P}_{max}=0.5$ dBA. The version of the neural model without any power constraint performs the best compared to the baseline (p-value $= 7\cdot 10^{-8}$). 
Applying a power constraint results in a decrease in performance all the more important the stricter the constraint (low $\Delta \mathcal{P}_{max}$), particularly in the low-frequency range to the point of becoming less performant than Estreder's PEQ. This outcome is expected, given the relatively low weight of high frequencies in the power measurement. 
When the power constraint is strict, the low and mid frequencies are more significantly affected. This trend is confirmed when examining the GLD. Without a power constraint, the neural model achieves excellent NMR performance by significantly amplifying the musical signal compared to Estreder's model. Adding the power constraint has then a clear beneficial effect on the GLD measure, thus achieving significantly better results compared to the baseline model, except at high frequencies where the model is less affected by the constraint. In particular, both neural models with constraints $\Delta \mathcal{P}_{max} =$ 2, 1 dBA achieve a better NMR than Estreder's model (p-value of $10^{-6}$ and 0.01) and a better broadband GLD (p-value of $1.5 \cdot 10^{-4}$ and $3.3 \cdot 10^{-9}$).
<!-- {: .text-justify} -->

## Earbuds impact

The noises in the test set are filtered with the frequency responses of 3 models of earbuds to reproduce their respective passive attenuations : 
- Bose headphones QuietComfort
- Sony earbuds WF-1000XM4 with sound isolating sleeves
- Apple Airpods with smooth tips
{: .text-justify}

![image-center](figures/earbuds_fr.png){: .align-center}

The Bose and Sony headphones act as low-pass filters while the Airpods have a much flatter effect.
{: .text-justify}

### NMR

![image-center](figures/earbuds_nmr.png){: .align-center}

### GLD

![image-center](figures/earbuds_gld.png){: .align-center}

The airpods clearly reduce less the noise than the two other headphones models. Therefore the initial NMR is greater in this case and the performances for all models are reduced. This can be explained by a closer look at examples of results: the system concentrates primarily on the Bark bands where the NMR is highest (generally mid and high frequencies in the case of airpods), even if this means leaving out other bands. 
{: .text-justify}

## Environments

The noise set is composed of samples from defined environments :
- Urban
- Transportation (train / plane / boat)
- Cocktail party (restaurant / café)
- Construction site
- Beach
- Indoor office
{: .text-justify} 

### NMR

![image-center](figures/environment_nmr.png){: .align-center}

### GLD

![image-center](figures/environment_gld.png){: .align-center}

The performance obtained is broadly as expected, with poorer results in terms of both NMR and GLD in the noisiest environments (all the more so as the music level is contained in the [45, 100] dBA range).
{: .text-justify} 

## SNR

We can also view the metrics by SNR band (in dB) between unprocessed music and noise.

### NMR

![image-center](figures/snr_nmr.png){: .align-center}

### GLD

![image-center](figures/snr_gld.png){: .align-center}
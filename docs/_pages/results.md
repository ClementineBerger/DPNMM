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

# Metrics

Two objective metrics are considered to perform the evaluation of the models. We compute a mean Noise-to-Mask Ratio (NMR) per audio sample of the test set, only selecting the Bark bands where the initial music masking threshold is below the noise level :

$$\text{NMR} = \frac{1}{M} \sum_{n, \nu} (1-m_\nu(n)) | P_{dB}^{noise}(n,\nu) - \hat{T}_{dB}(n,\nu) |,$$

where $M = \sum_{n, \nu} (1-m_\nu(n))$ with $m_\nu$ a mask such that $m_\nu(n) = 0$ if the initial threshold is below the noise, and $m_\nu(n) = 1$ otherwise. The obtained NMR is compared to the initial NMR with the unprocessed music to evaluate how much the system can improve the masking effect on the bands where it is required. However, the system may as well induce power variations in the other bands. To evaluate this effect we also compute a mean Global Level Difference (GLD) : 

$$\text{GLD} = \frac{1}{N} \sum | \hat{\mathcal{P}}_{dBA}^{music}(n) - \mathcal{P}_{dBA}^{music}(n) | .$$

Both metrics are computed by frequency ranges: broadband, first third of Bark bands (low), second third (medium), and last third (high). 

# Results


<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Affichage de deux images côte à côte</title>
    <style>
        /* Conteneur Flex pour les images */
        .image-container {
            display: flex; /* Utilisation de Flexbox pour disposer les images en ligne */
            align-items: flex-start; /* Alignement des images par le haut */
            gap: 10px; /* Espacement de 10px entre les images, ajustable selon vos préférences */
        }
        .image-container img {
            max-width: 100%; /* Assure que les images ne dépassent pas la largeur du conteneur parent */
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

# Generative Aversarial learning in Text Classification

The current research on  semi-supervised GAN approach to enhance state-of-the-art models for text classification in terms of few labeled  and some amount of unlabeled examples.

### Approaches:

- GAN-Transformer (BERT / distilBERT / distilRoBERTa / ALBERT) [1]
  - Conditional-GAN-Transformer
  - \+ Generator pretraining as decoder in autoencoder framework
  - \+ Negative Data Augmentation (NDA) [2]
  - \+ Manifold regularization [3]
  - \+ GAN disitillation [4]

### References


[1] D. Croce, G. Castellucci and R. Basili, "GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples", 2020

[2] S. Shayesteh and D. Inkpen, "Generative Adversarial Learning with Negative Data Augmentation for Semi-supervised Text Classification", 2022.

[3] B. Lecouat, C.-S. Foo, H. Zenati and V. R. Chandrasekhar, "Semi-Supervised Learning with GANs: Revisiting Manifold Regularization" 2018.

[4] M. Pennisi, S. Palazzo and C. Spampinato, "Self-improving classification performance through GAN distillation", 2021.

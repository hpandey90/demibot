# DemiBot : Chatter Bot <img src="logo.ico" align="right" />

## Synopsis

A chatbot developed using sequence to sequence model is capable of predicting next sentence based on previous sentence/conversations. We trained our model on movie dialogue corpus and twitter dataset to have a conversation.

## Language and Libraries

- Python : 3.5.2
- Tensorflow-GPU : 1.0.0
- Nltk : 3.2.2
- matplotlib : 2.0.0
- numpy : 1.12.0

## Hardware Used

NVIDIA GTX 1080 8GB for training

For training Attention Mechanism model it would need a minimum of 8 GB GPU but without Attention Mechanism 4GB would be suffice.

## Motivation

To create an agent that can have conversation like humans is one of the open area in machine learning which attracts pool of talent to solve and come up with a solution to the problem. To address this problem of open ended discussion we have developed a conversational model using machine learning which is generative in nature and is capable of having open conversations.

## Installation

Below are the things you need to do for playing with this model.
### Prepare
Download data
```
>> python getData.py
```
Download Model
```
>> python getModel.py
```
### Chatting
```
>> python demiBot.py
```
### Training
First get data and model then run following code :
```
>> python train.py
```
once training is done go to Chatting.

## Contributors and Support

- Ashish Katiyar - Grad Student at UF
- Braj Gopal Maity - Grad Student at UF
- Himanshu Pandey - Grad Student at UF


## License

MIT

## Special Thanks:

#### [Dr. Xiaolin (Andy) Li](http://www.andyli.ece.ufl.edu/index.php/people/principal-investigator/)

## References :

- http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
- http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- [Sordoni et al.](https://arxiv.org/pdf/1507.04808.pdf)
- [Li et al.](https://arxiv.org/pdf/1603.06155.pdf)

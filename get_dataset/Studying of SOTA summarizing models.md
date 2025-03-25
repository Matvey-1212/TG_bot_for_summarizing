# Studying of SOTA summarizing models

SOTA summarizing models refer to models that are at the current optimal level (State-of-the-Art) in the text summarization task.

#### BART

BART: An abbreviation for **Bidirectional Attention Representations from Transformers**. It is a Transformer-based sequence-to-sequence model that combines an encoder and a decoder architecture. During pre-training, BART uses various text noises such as deletion, insertion, and replacement to learn robust representations of text and can better capture the semantic and structural information of text when generating summaries.

##### Architectural Foundation

* **Transformer Core**: BART is based on the Transformer architecture. The core of the Transformer is the self - attention mechanism and the feed - forward neural network. The self - attention mechanism can calculate the degree of association between each position in the input sequence and other positions in parallel, efficiently capturing long - range dependencies in the text. The feed - forward neural network further extracts and transforms features from the output of the self - attention mechanism.
* **Encoder - Decoder Structure**: BART adopts an encoder - decoder architecture. The encoder is responsible for encoding the input text into a fixed - length vector representation, capturing the semantic information of the input text. The decoder, based on the output of the encoder and the previously generated output sequence, gradually generates the target text. Both the encoder and the decoder consist of multiple stacked Transformer layers. This structure enables the model to deeply understand the semantics of the text and generate it.

##### Pre - training Tasks

* **Denoising Auto - encoder**: The pre - training process of BART is mainly based on the idea of the denoising auto - encoder. It performs various noise - corrupting operations on the input text, such as randomly masking, deleting, or shuffling words or sentences in the text, and then the model is required to learn to restore the original text from these corrupted inputs. In this way, the model can learn the internal structure and semantic information of the text, improving its ability to understand various text phenomena.
* **Joint Training of Multiple Tasks**: During the pre - training process, BART also simultaneously trains on multiple natural language processing tasks, such as language modeling, text generation, question - answering, etc. Through the joint training of these tasks, the model can learn more general language knowledge and semantic representations, enabling it to perform well in various downstream tasks.

##### Attention Mechanism

* **Bidirectional Attention**: In the encoder, BART uses the bidirectional attention mechanism to simultaneously focus on the context information before and after the input text. This two - way attention method allows the model to better understand the context semantics of each word in the entire sentence or document, capturing richer semantic dependencies.
* **Cross - Attention**: In the decoder, in addition to the self - attention mechanism, BART also uses the cross - attention mechanism. The cross - attention mechanism is used to interact the current state of the decoder with the output of the encoder, enabling the decoder to make full use of the semantic information of the input text captured by the encoder when generating text, thus better generating reasonable outputs related to the input.

##### Word Embedding and Position Encoding

* **Word Embedding**: BART converts the words in the input text into low - dimensional vector representations, i.e., word embeddings. Word embeddings can encode the semantic information of words into the vector space, so that words with similar semantics are close in the vector space. The model will automatically learn these word embeddings during the training process, enabling it to better adapt to various natural language processing tasks.
* **Position Encoding**: Since the Transformer itself does not have a natural perception of the position information of the text sequence, BART uses position encoding to add position information to each word. The position encoding can be sinusoidal or learnable, which allows the model to perceive the order and position relationship of words when processing the text, thus better understanding the semantic structure of the text.

#### PEGASUS

PEGASUS: By introducing large-scale text data and specific pre-training tasks such as extractive summary generation and generative summary generation in pre-training, this model enables learning of different types of summary generation patterns. When generating summaries, PEGASUS can flexibly select an appropriate generation strategy according to the content and context of the input text.

##### Architectural Foundation

* **Based on Transformer**: Similar to many modern natural language processing models, PEGASUS is built on the Transformer architecture. The core advantage of the Transformer lies in its self - attention mechanism, which can compute the dependencies between each position in the input sequence and all other positions in parallel. This enables it to efficiently capture long - range semantic dependencies in the text, providing strong support for the model to understand the global semantics of the text.
* **Encoder - Decoder Structure**: PEGASUS adopts the classic encoder - decoder architecture. The encoder is responsible for encoding the input text into a low - dimensional vector representation, which contains the semantic information of the input text. The decoder, based on the output of the encoder and the part of the summary that has been generated, gradually generates the complete text summary. Both the encoder and the decoder consist of multiple stacked Transformer layers. This deep architecture allows the model to conduct multi - level and multi - perspective semantic analysis and generation of the text.

##### Pre - training Tasks

* **Gap Sentence Generation**: The main pre - training task of PEGASUS is Gap Sentence Generation. During pre - training, the model processes the input text by randomly selecting some consecutive sentences as "gaps", and then the model learns to predict these deleted sentences from the remaining text. In this way, the model can learn the coherence, semantic associations of the text, and how to generate reasonable content based on the context, thus improving the model's ability to understand the overall semantics of the text and generate relevant content.
* **Training on Large - scale Corpora**: PEGASUS is pre - trained on large - scale text corpora that cover texts from various fields and topics, such as news, novels, academic papers, etc. By training on diverse corpora, the model can learn the language styles, semantic patterns, and structural characteristics of different types of texts, thus having stronger generalization ability and performing well in various real - world text summarization tasks.

##### Attention Mechanisms

* **Encoder Self - Attention**: In the encoder, the self - attention mechanism is used to calculate the importance weights between each position in the input text and other positions. This enables the model to flexibly allocate attention according to the semantic relevance of different positions. For example, when processing a long document, the model can quickly capture the positions of key information through the self - attention mechanism and pay more attention to them, so as to better understand the core content of the text.
* **Decoder Cross - Attention and Self - Attention**: The cross - attention mechanism in the decoder is used to interact the current state of the decoder with the output of the encoder, so that the decoder can make full use of the semantic information of the input text when generating the summary. At the same time, the decoder also uses the self - attention mechanism to model the part of the summary that has been generated, capturing the semantic dependencies within the summary to ensure the coherence and logic of the generated summary.

##### Word Embedding and Position Encoding

* **Word Embedding**: PEGASUS maps the words in the input text into low - dimensional word vector representations, i.e., word embeddings. These word vectors can capture the semantic information of words, so that words with similar semantics have similar representations in the vector space. The model will automatically learn these word embeddings during pre - training, continuously adjusting the representation of word vectors to better meet the needs of the text summarization task.
* **Position Encoding**: Since the Transformer architecture itself is not sensitive to the sequential information of the text, PEGASUS uses position encoding to add position information to each word. The position encoding can be sinusoidal or learnable, which enables the model to perceive the order and position relationships of words in the text. Thus, when generating the summary, the model can correctly consider the context order of the text and generate a summary content that is more logical and semantic.

#### T5

T5: Abbreviation for Text-to-Text Transfer Transformer. It unifies all natural language processing tasks as text-to-text transformation problems. Through unsupervised pre-training on large-scale data and then fine-tuning on specific tasks. T5 can utilize its powerful language understanding and generation capabilities in the text summarization task to generate high-quality summaries.

##### Core Architecture

* **Based on Transformer**: T5 is built upon the Transformer architecture. The core of the Transformer is the self - attention mechanism, which can compute the dependencies between positions in the input sequence in parallel, efficiently capturing long - range semantic dependencies. For example, when dealing with a long novel, it can quickly identify the relationships between plot elements in different chapters, enabling the model to understand the global semantics of the text.
* **Encoder - Decoder Structure**: It adopts the classic encoder - decoder architecture. The encoder encodes the input text into a low - dimensional vector representation that contains the semantic information of the input. The decoder, based on the output of the encoder and the content that has been generated, gradually generates the target text. In machine translation, the encoder processes the source - language sentence, and the decoder generates the target - language sentence.

##### Pre - training Tasks

* **Text - to - Text Task Format**: T5 unifies all natural language processing tasks into the text - to - text format. Whether it is text classification, question - answering, or summarization, these tasks are regarded as conversions from one form of text to another. For instance, in a text - classification task, the input text can be transformed into a text of class labels, and in a question - answering task, the question text and relevant context are transformed into an answer text.
* **Large - scale Multi - task Pre - training**: T5 is pre - trained on a vast amount of multi - task data from diverse sources, covering various domains and task types. Through this large - scale multi - task pre - training, the model can learn the language patterns and semantic characteristics of different tasks, enhancing its generalization ability and adaptability to various natural language processing tasks.

##### Attention Mechanisms

* **Multi - head Attention Mechanism**: The attention mechanism in T5 employs the multi - head attention mechanism. That is, through parallel computations of multiple "heads", it captures semantic information from different perspectives and levels. Each head focuses on different parts of the input, and when combined, they can comprehensively understand the text semantics. For example, when dealing with complex legal text, different heads can respectively focus on legal subjects, legal actions, legal consequences, etc.
* **Self - Attention and Cross - Attention**: The self - attention mechanism is used in the encoder to calculate the weights of positions within the input text. In the decoder, there is not only the self - attention mechanism to handle the dependencies of the generated text but also the cross - attention mechanism to integrate the output information of the encoder, ensuring the relevance and coherence between the generated text and the input information.

##### Word Embedding and Position Encoding

* **Word Embedding**: Words in the input text are mapped into low - dimensional word vectors, which capture the semantic information of words, making words with similar semantics close in the vector space. The model automatically learns word embeddings during pre - training and continuously adjusts them to adapt to various tasks.
* **Position Encoding**: To compensate for the Transformer's insensitivity to text sequential information, T5 uses position encoding to add position information to each word, helping the model perceive the order and position relationships of words, so that the generated text conforms to logical and semantic sequences.

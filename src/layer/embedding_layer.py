from keras.layers import Embedding


def embedding_layer(vocabulary_size, embedding_matrix, embedding_dim, max_sentence_length):
    """

    Args:
        vocabulary_size:
        embedding_matrix:
        embedding_dim:
        max_sentence_length:

    Returns:

    """
    return Embedding(vocabulary_size,
                     embedding_dim,
                     weights=[embedding_matrix],
                     input_length=max_sentence_length,
                     trainable=False)

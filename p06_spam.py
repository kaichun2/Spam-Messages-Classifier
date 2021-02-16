import collections

import numpy as np

import util
import svm

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    str_list = message.split(' ')
    for i in range(len(str_list)):
        str_list[i] = str_list[i].lower()
    return str_list
    # *** END CODE HERE ***

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # count all the word occurance 
    msg_occur = {}
    diction = {}
    total_word_in_dict = 0
    # loop through each message
    for i in range(len(messages)):
        cur_line = set(get_words(messages[i]))
        for j in cur_line:
            if j in msg_occur:
                msg_occur[j] += 1
                # add word into dict
                if msg_occur[j] == 5:
                    diction[j] = total_word_in_dict
                    total_word_in_dict += 1
            else:
                msg_occur[j] = 1
    return diction
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    result = np.zeros((len(messages), len(word_dictionary)))
    for i in range(len(messages)):
        cur_str = get_words(messages[i])
        for j in range(len(cur_str)):
            if cur_str[j] in word_dictionary:
                result[i][word_dictionary[cur_str[j]]] += 1
    return result
    # *** END CODE HERE ***

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # each column => a word, last row is total count of y
    # each row => each y value, row 1 => y = 0, row 2 => y = 1
    model = np.zeros((2, matrix.shape[1] + 1))
    for i in range(matrix.shape[0]):
        # for each message
        total_word_in_this_msg = 0
        for j in range(matrix.shape[1]):
            if labels[i] == 0:
                model[0][j] += matrix[i][j]
            else:
                model[1][j] += matrix[i][j]
            total_word_in_this_msg += matrix[i][j]
        if labels[i] == 0:
            model[0][matrix.shape[1]] += total_word_in_this_msg
        else: 
            model[1][matrix.shape[1]] += total_word_in_this_msg
    # smoothing
    for i in range(matrix.shape[1]):
        model[0][i] += 1
        model[1][i] += 1
        model[1][matrix.shape[1]] += 1
        model[0][matrix.shape[1]] += 1
    # change all entries into P(xj = k | y = 0 or 1)
    # change last entry into P(y = 0 or 1)
    for i in range(matrix.shape[1]):
        model[0][i] /= model[0][matrix.shape[1]]
        model[1][i] /= model[1][matrix.shape[1]]
    total_y = model[0][matrix.shape[1]] + model[1][matrix.shape[1]]
    model[0][matrix.shape[1]] /= total_y
    model[1][matrix.shape[1]] /= total_y
    return model
    # *** END CODE HERE ***

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """
    # *** START CODE HERE ***
    result = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        condiprob_y1 = 0
        condiprob_y0 = 0
        # for each message
        for j in range(matrix.shape[1]):
            condiprob_y0 += np.log(model[0][j]) * matrix[i][j]
            condiprob_y1 += np.log(model[1][j]) * matrix[i][j]
        # Mutilply by P(y = 0 or 1)
        condiprob_y0 += np.log(model[0][matrix.shape[1]])
        condiprob_y1 += np.log(model[1][matrix.shape[1]])
        if condiprob_y1 > condiprob_y0:
            result[i] = 1
    return result
    # *** END CODE HERE ***

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    result = {}
    arr_min = 0
    arr_min_word = ''
    for cur_word in dictionary:
        cur_word_ID = dictionary[cur_word]
        prob_y0 = model[0][cur_word_ID]
        prob_y1 = model[1][cur_word_ID]
        cur_prob = np.log(prob_y1 / prob_y0)
        # add words if less than 5 elements in result
        if len(result) < 5:
            result[cur_word] = cur_prob
            if len(result) == 1:
                arr_min = cur_prob
                arr_min_word = cur_word
            else:
                if arr_min > cur_prob:
                    arr_min = cur_prob
                    arr_min_word = cur_word
        else:
            if np.log(prob_y1 / prob_y0) > arr_min:
                del result[arr_min_word]
                result[cur_word] = cur_prob
                arr_min = cur_prob
                arr_min_word = cur_word
                # i is key, find min in array again
                for i in result:
                    if arr_min > result[i]:
                        arr_min = result[i]
                        arr_min_word = i
    indic_word = []
    # last one is the arr_min_word found from above
    while len(result) != 0:
        cur_max = arr_min
        cur_max_word = arr_min_word
        for i in result:
            if result[i] >= cur_max:
                cur_max = result[i]
                cur_max_word = i
        indic_word.append(cur_max_word)
        del result[cur_max_word]
    return indic_word
    # *** END CODE HERE ***

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    opt_radius = 0
    accuracy = 0
    # *** START CODE HERE ***
    for i in range(len(radius_to_consider)):
        output = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius_to_consider[i])
        correct = 0
        for j in range(len(val_labels)):
            if val_labels[j] == output[j]:
                correct += 1
        if accuracy < (correct / len(val_labels)):
            opt_radius = radius_to_consider[i]
            accuracy = correct / len(val_labels)
    return opt_radius
    # *** END CODE HERE ***

def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)


    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()
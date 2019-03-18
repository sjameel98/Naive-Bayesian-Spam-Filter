import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.
    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here

    spam = file_lists[0]
    ham = file_lists[1]

    vocab = util.get_counts(ham+spam).keys()
    qdvalues = {}
    pdvalues = {}
    spamcounts = util.get_counts(spam)
    hamcounts = util.get_counts(ham)

    Ns = len(spam)
    Nh = len(ham)

    for word in vocab:
        pdvalues[word] = (spamcounts[word]+1)/(Ns+2)
        qdvalues[word] = (hamcounts[word]+1)/(Nh+2)
    probabilities_by_category = (pdvalues, qdvalues)

    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, eps = 1):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    
    probabilities_by_category: output of function learn_distributions
    
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    #eps is our adjustment
    #our priors
    Pspam = np.log(prior_by_category[0])
    Pham = np.log(prior_by_category[1])

    #all our vocabulary
    vocab = probabilities_by_category[0].keys()

    #words in our email
    words = util.get_words_in_file(filename)

    for word in vocab:
        #depending on if word is in the email or not, add the log probability of it being in or out
        if word in words:
            Pspam += np.log(probabilities_by_category[0][word])
            Pham += np.log(probabilities_by_category[1][word])
        else:
            Pspam += np.log(1-probabilities_by_category[0][word])
            Pham += np.log(1-probabilities_by_category[1][word])
    if Pspam > eps*Pham:
        classify_result = ('spam', [Pspam, Pham])
    else:
        classify_result = ('ham', [Pspam, Pham])
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    type1 = []
    type2 = []
    for i in np.linspace(0.7, 1.3, 20):
        performance_measures = np.zeros([2, 2])
        for email in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(email,
                                                      probabilities_by_category,
                                                      priors_by_category, i)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(email)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        #Type 1 Error and Type 2 Error respectively
        type1.append(totals[0] - correct[0])
        type2.append(totals[1] - correct[1])
    plt.plot(np.array(type1), np.array(type2), marker = 'x', markerfacecolor = 'blue', linestyle = '--')
    plt.xlabel('Type 1 Error')
    plt.ylabel('Type 2 Error')
    plt.title('Tradeoff Curve')
    plt.savefig('nbc.png')
    plt.show()

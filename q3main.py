import numpy as np
import pandas as pd

# Load the dataset
def load_data():
    y_train = pd.read_csv('y_train.csv', header=None, delim_whitespace=True)
    y_test = pd.read_csv('y_test.csv', header=None, delim_whitespace=True)
    x_train = pd.read_csv('X_train.csv', delim_whitespace=True, low_memory=False)
    x_test = pd.read_csv('X_test.csv', delim_whitespace=True, low_memory=False)
    
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)
    
    return train_df, test_df

# Train Multinomial Naive Bayes without smoothing
def train_multinomial_nb(train_df):
    # Separate features and the labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    # prior probabilities
    class_counts = y_train.value_counts().sort_index()
    total_documents = len(y_train)
    priors = class_counts / total_documents

    # likelihoods without smoothing
    likelihoods = {}
    for label in np.unique(y_train):
        word_counts = X_train[y_train == label].sum(axis=0)
        likelihoods[label] = word_counts / word_counts.sum()

    return priors, likelihoods

# Train Multinomial Naive Bayes with smoothing
def train_multinomial_nb_with_smoothing(train_df, alpha=1):
    # Separate the features and the labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    # prior probabilities
    class_counts = y_train.value_counts().sort_index()
    total_documents = len(y_train)
    priors = class_counts / total_documents

    # likelihoods with smoothing
    likelihoods = {}
    for label in priors.index:
        label_feature_counts = X_train[y_train == label].sum(axis=0) + alpha
        total_label_feature_counts = label_feature_counts.sum()
        likelihoods[label] = label_feature_counts / total_label_feature_counts

    return priors, likelihoods

# Train Bernoulli  with smoothing
def train_bernoulli_nb_with_smoothing(train_df, alpha=1):
    # Separate the features and the labels
    X_train = train_df.iloc[:, :-1].astype(bool).astype(int)  # Ensure the features are binary
    y_train = train_df.iloc[:, -1]

    # Calculate prior probabilities
    class_counts = y_train.value_counts().sort_index()
    total_documents = len(y_train)
    priors = class_counts / total_documents

    #likelihoods with smoothing
    likelihoods = {}
    for label in priors.index:
        label_feature_counts = X_train[y_train == label].sum(axis=0) + alpha
        label_feature_absence_counts = (X_train[y_train == label] == 0).sum(axis=0) + alpha
        
        # Probabilities for word presence and absence
        total_label_documents = class_counts[label] + 2 * alpha
        likelihoods[label] = {
            'presence': label_feature_counts / total_label_documents,
            'absence': label_feature_absence_counts / total_label_documents
        }

    return priors, likelihoods

def predict_multinomial_nb(test_df, priors, likelihoods, log0_value):
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    # Log probabilities for numerical stability
    log_priors = np.log(priors)
    log_likelihoods = {label: np.log(likelihood.replace(0, log0_value)) for label, likelihood in likelihoods.items()}

    # Array to hold the probabilities for each class
    probs = np.zeros((X_test.shape[0], len(priors)))

    # Compute probability for each class
    for index, (label, log_prior) in enumerate(log_priors.items()):
        probs[:, index] = X_test.dot(log_likelihoods[label]) + log_prior
        
    # Choose the highest probability
    predictions = np.argmax(probs, axis=1)
    
    return predictions


def predict_bernoulli_nb(test_df, priors, likelihoods):
    X_test = test_df.iloc[:, :-1].astype(bool).astype(int)  # Ensure the features are binary!!!
    y_pred = []

    # Iterate through each test example
    for index, test_example in X_test.iterrows():
        class_probabilities = {}
        # probability for each class
        for label, class_prior in priors.items():
            # log of the prior probability of the class
            log_prob = np.log(class_prior)
            # Add the log probability
            log_prob += (test_example * np.log(likelihoods[label]['presence']) + (1 - test_example) * np.log(likelihoods[label]['absence'])).sum()
            class_probabilities[label] = log_prob
        # Choose  highest probability
        y_pred.append(max(class_probabilities, key=class_probabilities.get))

    return y_pred

def calculate_accuracy(predictions, labels):
    return (predictions == labels).mean()

def calculate_confusion_matrix(predictions, labels):
    return pd.crosstab(labels, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)



def main():
    train_df, test_df = load_data()

    while True:
        choice = input("Press 1 to run Multinomial NB without smoothing with log(0) = -inf\n"
                       "Press 2 to run Multinomial NB without smoothing with log(0) = 10^-12\n"
                       "Press 3 to run Multinomial NB with smoothing\n"
                       "Press 4 to run Bernoulli NB with smoothing\n"
                       "Press Q to quit\n"
                       "Enter your choice: ")

        if choice == 'Q' or choice == 'q':
            print("Exiting the program.")
            break

        if choice == '1':
            priors, likelihoods = train_multinomial_nb(train_df)
            predictions = predict_multinomial_nb(test_df, priors, likelihoods, -np.inf)
            classifier_name = "Multinomial NB without smoothing with log(0) = -inf"
        elif choice == '2':
            priors, likelihoods = train_multinomial_nb(train_df)
            predictions = predict_multinomial_nb(test_df, priors, likelihoods, 1e-12)
            classifier_name = "Multinomial NB without smoothing with log(0) = 10^-12"
        elif choice == '3':
            priors, likelihoods = train_multinomial_nb_with_smoothing(train_df)
            predictions = predict_multinomial_nb(test_df, priors, likelihoods, 1e-12)
            classifier_name = "Multinomial NB with smoothing"
        elif choice == '4':
            priors, likelihoods = train_bernoulli_nb_with_smoothing(train_df)
            predictions = predict_bernoulli_nb(test_df, priors, likelihoods)
            classifier_name = "Bernoulli NB with smoothing"
        else:
            print("Invalid choice. Please try again.")
            continue

        accuracy = calculate_accuracy(predictions, test_df.iloc[:, -1])
        conf_matrix = calculate_confusion_matrix(predictions, test_df.iloc[:, -1])

        print(f"\n{classifier_name} Accuracy: {accuracy:.3f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()





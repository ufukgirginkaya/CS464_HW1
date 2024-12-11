This program is a Python script for running different Naive Bayes classifiers on a dataset. The script includes Multinomial Naive Bayes with and without smoothing and Bernoulli Naive Bayes with smoothing. The script will prompt the user to select which classifier to run and display the resulting accuracy and confusion matrix.

REQUIREMENTS:
- Python 3.x
- pandas library
- numpy library

To run the script, you will need a dataset split into four files: 'X_train.csv', 'X_test.csv', 'y_train.csv', and 'y_test.csv'. These files should be in the same directory as the script.

INSTRUCTIONS:
1. Ensure you have the required Python version and libraries installed.
2. Place your dataset files in the same directory as the script.
3. Run the script in your terminal or command prompt.
4. You will be prompted to choose which classifier you want to run:
   - Press 1 for Multinomial NB without smoothing with log(0) = -inf
   - Press 2 for Multinomial NB without smoothing with log(0) = 10^-12
   - Press 3 for Multinomial NB with smoothing
   - Press 4 for Bernoulli NB with smoothing
   - Press Q to quit the program
5. After making a selection, the classifier will run, and the accuracy and confusion matrix will be displayed.
6. Once the results are displayed, you will be prompted again to run another classifier or exit the program.

If you encounter any issues or have questions, please refer to the contact information below.

CONTACT INFORMATION:
Ufuk Girginkaya
ufuk.girginkaya@ug.bilkent.edu.tr
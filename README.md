# ML twitter decider

Given a dataset of tweets posted on twitter by certain peoples, learner predicts which person tweeted what. 

### Learner
Many learners were tested, such as Random Forest, KNN, Logistic Regression, SVM, Naive Bayes, with and without cross validation.
In the end, the chosen learner was SVM with SGD training (sk-learn implementation) and achieves accuracy of 0.84+.
The data is pre-processed with CountVectorizer and TfidfTransformer in a Bag of Words representation after it
was 'cleaned' to achieve the best result.

### Data
The learner was provided with the following 10 celebrities: 
Donald Trump, Joe Biden, Conan O'brien, Ellen Degeneres, Kim Kardashian, Lebron James,
 Lady Gaga, Cristiano Ronaldo, Jimmy Kimmel and Arnold Schwarzenegger.

Data was provided by HUJI IML staff.

Learner was built as a team effort in a HUJI Hackathon.

### Classification report
Different scoring between all classes
![Classification report for the above celebrities](https://github.com/luko007/ML_twitter_celebs/blob/master/test_plot_classif_report.png)


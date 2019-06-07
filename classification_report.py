import itertools
import matplotlib as plt
import numpy as np

# Plots a static all-feature-heatmap connecting scores between all features
def plot_classification_report(title='Classification report',
                               cmap='RdBu'):
    classificationReport = """                        precision    recall  f1-score   support
         Donald Trump       0.83      0.92      0.87       579
            Joe Biden       0.86      0.86      0.86       339
        Conan O'brien       0.82      0.82      0.82       611
      Ellen Degeneres       0.79      0.88      0.83       655
       Kim Kardashian       0.76      0.84      0.80       646
         Lebron James       0.88      0.82      0.85       691
            Lady Gaga       0.85      0.75      0.80       621
    Cristiano Ronaldo       0.87      0.86      0.86       640
         Jimmy kimmel       0.85      0.78      0.82       637
Arnold schwarzenegger       0.88      0.84      0.86       637
             accuracy                           0.84      6056
            macro avg       0.84      0.84      0.84      6056
         weighted avg       0.84      0.84      0.84      6056"""

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0] +' '+ t[1])
        v = [x for x in t[2: len(t) - 1]]
        support.append(t[-1])
        class_names.append(t[0] +' '+ t[1])
        if v:
            plotMat.append([float(x) for x in v])

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Classes')
    plt.xlabel('Accuracy')
    plt.tight_layout()

def main_plot_class_rep(classificationReport):
    plot_classification_report (classificationReport)
    plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')


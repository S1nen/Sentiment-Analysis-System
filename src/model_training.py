from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import x_train_scaled,y_train,x_test_scaled,y_test
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

rfc=RandomForestClassifier()
rfc.fit(x_train_scaled,y_train)

print("Accuracy on training data:",rfc.score(x_train_scaled,y_train))
print("Accuracy on testing data:",rfc.score(x_test_scaled,y_test))

#found accuracy on trainig data-0.99
#found accuracy on test data-0.93

y_pred=rfc.predict(x_test_scaled)
cm=confusion_matrix(y_test,y_pred)

cmdisplay=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rfc.classes_)
cmdisplay.plot()
plt.show()

accuracy=cross_val_score(estimator=rfc,X=x_train_scaled,y=y_train,cv=10)
print("Accuracy:",accuracy.mean())
print("standard variance:",accuracy.std())

#found standard variance-0.0088
#found mean accuracy-0.93

#grid search for best parameters in random forest

parameters={
    'bootstrap':[True],
    'max_depth':[20,40],
    'min_samples_split':[8,12],
    'n_estimators':[30,70]
}

cv_object=StratifiedKFold(n_splits=2)
gridsearch=GridSearchCV(estimator=rfc,param_grid=parameters,cv=cv_object,verbose=2,return_train_score=True)
gridsearch.fit(x_train_scaled,y_train.ravel())
print("Best parameters",gridsearch.best_params_)

#found best parameters are {'bootstrap': True, 'max_depth': 40, 'min_samples_split': 12, 'n_estimators': 70}

#training with the best params-

best_rfc=RandomForestClassifier(bootstrap=True,max_depth=40,min_samples_split=12,n_estimators=70,random_state=42,class_weight='balanced')
best_rfc.fit(x_train_scaled,y_train)
y_pred2=best_rfc.predict(x_test_scaled)
print("accuracy on training data after parameter tuning:",best_rfc.score(x_train_scaled,y_train))
print("accuracy on test data after parameter tuning:",best_rfc.score(x_test_scaled,y_test))

#found accurcay on training data after parameter tuning-0.95
#found accuracy on test data after parameter tuning-0.92

y_pred2=best_rfc.predict(x_test_scaled)
cm2=confusion_matrix(y_test,y_pred2)

cmdisplay2=ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=best_rfc.classes_)
cmdisplay2.plot()
plt.show()

print("classification report:",classification_report(y_test,y_pred2))


# Import LabelEncoder
from sklearn import preprocessing
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

rainy_orNot = ['No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', ' No', 'No', 'Yes', 'No',
               'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', ' No', 'Yes', 'Yes', 'No',
               'No', 'No', 'Yes', 'No', 'Yes']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild',
        'Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Cool', 'Cool', 'Hot', 'Hot',
        'Hot', 'Hot']
play = ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']

# create labelEncoder
le = preprocessing.LabelEncoder()

rainy_orNot_encoded = le.fit_transform(rainy_orNot)
temp_encoded = le.fit_transform(temp)
play_encoded = le.fit_transform(play)

label = le.fit_transform(play)
print(rainy_orNot_encoded)
print(temp_encoded)
print(play_encoded)
print(label)

feature = []
for i in range(30):
    feature.append((rainy_orNot_encoded[i], temp_encoded[i], play_encoded[i]))

# Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(feature, label)
# Predict Output
predicted = model.predict([[1, 1, 1]])
print(predicted) # 1: Play

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e690983",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4318bc8",
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bd39b8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('omw-1.4')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e847e00e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "import json\n",
    "import pickle\n",
    "\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Activation, Dropout\n",
    "from tensorflow.keras.optimizers import SGD\n",
    "import random\n",
    "\n",
    "words = []\n",
    "classes = []\n",
    "documents = []\n",
    "ignore_words = ['?', '!']\n",
    "data_file = open('intents.json').read()\n",
    "intents = json.loads(data_file)\n",
    "\n",
    "for intent in intents['intents']:\n",
    "    for pattern in intent['patterns']:\n",
    "\n",
    "        # tokenize each word\n",
    "        w = nltk.word_tokenize(pattern)\n",
    "        words.extend(w)\n",
    "        # add documents in the corpus\n",
    "        documents.append((w, intent['tag']))\n",
    "\n",
    "        # add to our classes list\n",
    "        if intent['tag'] not in classes:\n",
    "            classes.append(intent['tag'])\n",
    "\n",
    "# lemmaztize and lower each word and remove duplicates\n",
    "words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]\n",
    "words = sorted(list(set(words)))\n",
    "# sort classes\n",
    "classes = sorted(list(set(classes)))\n",
    "# documents = combination between patterns and intents\n",
    "print(len(documents), \"documents\")\n",
    "# classes = intents\n",
    "print(len(classes), \"classes\", classes)\n",
    "# words = all words, vocabulary\n",
    "print(len(words), \"unique lemmatized words\", words)\n",
    "\n",
    "pickle.dump(words, open('words.pkl', 'wb'))\n",
    "pickle.dump(classes, open('classes.pkl', 'wb'))\n",
    "\n",
    "# create our training data\n",
    "training = []\n",
    "# create an empty array for our output\n",
    "output_empty = [0] * len(classes)\n",
    "# training set, bag of words for each sentence\n",
    "for doc in documents:\n",
    "    # initialize our bag of words\n",
    "    bag = []\n",
    "    # list of tokenized words for the pattern\n",
    "    pattern_words = doc[0]\n",
    "    # lemmatize each word - create base word, in attempt to represent related words\n",
    "    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]\n",
    "    # create our bag of words array with 1, if word match found in current pattern\n",
    "    for w in words:\n",
    "        bag.append(1) if w in pattern_words else bag.append(0)\n",
    "\n",
    "    # output is a '0' for each tag and '1' for current tag (for each pattern)\n",
    "    output_row = list(output_empty)\n",
    "    output_row[classes.index(doc[1])] = 1\n",
    "\n",
    "    training.append([bag, output_row])\n",
    "# shuffle our features and turn into np.array\n",
    "random.shuffle(training)\n",
    "training = np.array(training)\n",
    "# create train and test lists. X - patterns, Y - intents\n",
    "train_x = list(training[:, 0])\n",
    "train_y = list(training[:, 1])\n",
    "print(\"Training data created\")\n",
    "\n",
    "# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons\n",
    "# equal to number of intents to predict output intent with softmax\n",
    "model = Sequential()\n",
    "model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(len(train_y[0]), activation='softmax'))\n",
    "\n",
    "# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model\n",
    "sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)\n",
    "model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])\n",
    "\n",
    "# fitting and saving the model\n",
    "hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)\n",
    "model.save('chatbot_model.h5', hist)\n",
    "\n",
    "print(\"model created and saved\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from exchangelib import DELEGATE, Account, Credentials, Message, Mailbox
from bs4 import BeautifulSoup
from markdown import markdown
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import pickle
import json

class MailFilter:

    def __init__(self):
        self._csv_fname = "./data/mail_exported.CSV"
        self._model_fname = "./data/spamModel"
        #self._sequences_len = 0
        self._tokenizer_fname = "./data/token.pickle"
        self._json_fname = "./data/cred.json"

        self._folder_spam = "unwichtig"
        self._folder_ham = "wichtig"

    def conn_init(self):
        with open(self._json_fname) as f:
            js = json.load(f)

        self._credentials = Credentials(
            username=js["username"],
            password=js["password"]
        )
        self._account = Account(
            primary_smtp_address=js["mail"],
            credentials=self._credentials,
            autodiscover=True,
            access_type=DELEGATE
        )

        self._credentials_ai = Credentials(
            username=js["username-ai"],
            password=js["password-ai"]
        )
        self._account_ai = Account(
            primary_smtp_address=js["mail-ai"],
            credentials=self._credentials_ai,
            autodiscover=True,
            access_type=DELEGATE
        )

        self._credentials_main = Credentials(
            username=js["username-main"],
            password=js["password-main"]
        )
        self._account_main = Account(
            primary_smtp_address=js["mail-main"],
            credentials=self._credentials_ai,
            autodiscover=True,
            access_type=DELEGATE
        )

        self._recipient = js["recipient"]

    def test_conn(self):

        for item in self._account.inbox.all().order_by('-datetime_received')[:10]:
            html = markdown(item.body)
            body = ''.join(BeautifulSoup(html, features="lxml").getText())
            print(item.subject, body)

    def create_csv(self):

        labelList = []
        textList = []
        cnt = 0
        for item in self._account.inbox.all().order_by('-datetime_received'):
            html = markdown(item.body)
            body = ''.join(BeautifulSoup(html, features="lxml").getText())
            mailText = body.strip().replace("\n", " ")

            if item.subject.startswith('###spam###') or item.subject.startswith('###unwichtig###')[:]:
                mailLabeL = "spam"
            else:
                mailLabeL = "ham"
            labelList.append(mailLabeL)
            textList.append(re.sub(r'^.*?Von:', 'Von:', mailText))
            print(cnt)
            cnt = cnt+1

        #df = pd.DataFrame(list(zip(labelList, textList)), columns=['label', 'text'])
        #df.to_csv(self._csv_name)

    def train(self):

        data = pd.read_csv(self._csv_fname)
        data["Label"] = data["Betreff"].str.contains("###wichtig###")
        data['BLabel'] = data['Label'].astype(int)
        data['Text'] = data['Text'].astype(str)
        #data['Text'] = data['Text'].str.replace("\r", " ")

        x = data['Text']
        y = data['BLabel']

        train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2)

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000)
        tokenizer.fit_on_texts(x)

        with open(self._tokenizer_fname, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Unique:", len(tokenizer.word_index))

        train_sequences = tokenizer.texts_to_sequences(train_data)
        test_sequences = tokenizer.texts_to_sequences(test_data)

        train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences)

        test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=train_padded.shape[1])

        train_padded = np.array(train_padded)
        test_padded = np.array(test_padded)
        train_label = np.array(train_label)
        test_label = np.array(test_label)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 20))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(train_padded, train_label, batch_size=8, epochs=20)
        test_loss, test_acc = model.evaluate(test_padded, test_label)
        print('Test accuracy:', test_acc)

        tf.keras.models.save_model(model, self._model_fname)

    def test(self):

        model = tf.keras.models.load_model(self._model_fname)
        with open(self._tokenizer_fname, 'rb') as handle:
            tokenizer = pickle.load(handle)

        for item in self._account.inbox.all().order_by('-datetime_received')[:100]:
            html = markdown(item.body)
            body = ''.join(BeautifulSoup(html, features="lxml").getText())
            email = body.strip().replace("\n", " ")

            email_sequence = tokenizer.texts_to_sequences([email])
            email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_sequence)
            email_padded = np.array(email_padded)


            prediction = model.predict(email_padded)
            print(item.subject)
            print("Wurde vom KNN erkannt als (Wahrscheinlichkeiten):")
            print(prediction)
            if np.argmax(prediction) == 1:
                print("x"*50)

    def apply(self):

        #print("start apply...")
        model = tf.keras.models.load_model(self._model_fname)
        with open(self._tokenizer_fname, 'rb') as handle:
            tokenizer = pickle.load(handle)

        try:
            for item in self._account_ai.inbox.all().order_by('-datetime_received')[:]:
                print("iterate mails...")

                html = markdown(item.body)
                body = ''.join(BeautifulSoup(html, features="lxml").getText())
                email = body.strip().replace("\n", " ")

                email_sequence = tokenizer.texts_to_sequences([email])
                email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_sequence)
                email_padded = np.array(email_padded)

                prediction = model.predict(email_padded)
                print(item.subject)
                print("Wurde vom KNN erkannt als (Wahrscheinlichkeiten):")
                print(prediction)

                print("sending mail...")
                m = Message(
                    account=self._account_ai,
                    subject= "###" + str(np.argmax(prediction))+ "### " + item.subject,
                    body=item.body,
                    to_recipients=[
                        Mailbox(email_address=self._recipient)
                    ]
                )
                m.send()
                item.delete()
                print("mail sent...")


        except Exception:
            print("no mails...")

        #print("stop apply...")

    def apply_to_account(self):

        model = tf.keras.models.load_model(self._model_fname)
        with open(self._tokenizer_fname, 'rb') as handle:
            tokenizer = pickle.load(handle)

        try:
            for item in self._account_main.inbox.all().order_by('-datetime_received')[:]:
                print("iterate mails...")

                html = markdown(item.body)
                body = ''.join(BeautifulSoup(html, features="lxml").getText())
                email = body.strip().replace("\n", " ")

                email_sequence = tokenizer.texts_to_sequences([email])
                email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_sequence)
                email_padded = np.array(email_padded)

                prediction = model.predict(email_padded)
                print(item.subject)
                print("Wurde vom KNN erkannt als (Wahrscheinlichkeiten):")
                print(prediction)

                if np.argmax(prediction) == 1:
                    folder = self._account_ai.inbox / self._folder_ham
                else:
                    folder = self._account_ai.inbox / self._folder_spam

                item.move(folder)
                item.refresh()


        except Exception:
            print("no mails...")

# MailFilter
### Mail classification with tensorflow and MS Exchange Server

## workflow
You need two mail accounts, here office@mail.com (your working account) and filter@mail.com (collect labeled data).

### 1. collect train and test data
Configure quicksteps in MS Outlook (office@mail.com) to change mail subjects form "bla bla bla" to "###ham### bla ..." and forward it to a new mail address (filter@mail.com). Do the same for spam.

### 2. generate csv file
Use Outlook to export mail data (filter@mail.com) as csv file.

### 3. train the ai
Use ***train()*** function to train your ai with tensorflow and store the model.

### 4. apply
Use ***apply_to_account()*** to classify mails (office@mail.com) and move them in "ham" or "spam" subfolders in your inbox (create these folders manually in your inbox).

___________________________________________________________________________________________

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

(c) 2021, Metin Karatas (m.karatas@sbs-herzogenaurach.de)


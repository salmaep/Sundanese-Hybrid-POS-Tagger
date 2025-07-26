from __future__ import division
from collections import Counter
import math
import re
from collections import Counter
from collections import defaultdict
import sys
from io import StringIO

# POS -> POS transition probabilities
# POS -> Word emission probabilities

# Kelas untuk mengalihkan output ke string dan file sekaligus
class TeeStringIO(StringIO):
    def __init__(self, filename, mode='w', encoding='utf-8'):
        super(TeeStringIO, self).__init__()
        self.file = open(filename, mode, encoding=encoding)
        self.original_stdout = sys.stdout
        
    def write(self, s):
        self.original_stdout.write(s)
        self.file.write(s)
        super(TeeStringIO, self).write(s)
        
    def close(self):
        self.file.close()
        super(TeeStringIO, self).close()

# Mulai merekam output
log_file = "hasil/hmm/pos_tagging_log.txt"
output_capture = TeeStringIO(log_file)
sys.stdout = output_capture

def split_punctuation(text):
    pattern = r"""
        (?:
            # \b(ka)(-)(\d+)\b                         # ✅ Special case: ka - 5
            # | 
            \b(?:HR|QS|M|E|R|S|H)\.                          # HR. atau QS.
            | [A-Z][a-z]{1,3}\.                      # Prof., Dr., Mr., dll
            | [A-Za-z]+(?:\.[A-Za-z]+)+              # QS.A.S., A.S., dkk
            | \b[A-Z]{2,5}\$                         # ✅ Gabungan simbol mata uang, contoh: US$
            | \d+(?:[.,]\d+)*                        # Angka desimal
            | \b[\w]+(?:[’'-][\w]+)+\b               # Kata dengan ’ atau - di tengah
            | \bSWT\b                                # SWT
            | [\w]+                                  # Kata biasa
            | ’(?=\s|$)                              # Apostrof sendiri
            | [“”"‘’',.!?;:()\[\]{}&/%-]             # Tanda baca umum
            | \S                                     # Karakter lainnya
        )
    """
    tokens = []
    for match in re.finditer(pattern, text, re.VERBOSE):
        if match.groups() and any(match.groups()):
            tokens.extend(g for g in match.groups() if g)
        else:
            tokens.append(match.group())
    return tokens


# make a map for context, emit, and transition
context = dict()
emit = dict()
transition = dict()

# Format training data: word/TAG
filename = 'dataset/Korpus_train.txt'

previous = "<s>"
endOfLine = "</s>"
context[previous] = 0
context[endOfLine] = 0

listOfTags = []

strOfWords = ""

print ('Start')

numOfUniqueWords = 3704


# Read the file for the 1st time
with open(filename, encoding='utf-8') as f:
	data = f.read().replace('\n', ' ')
	wordsAndTags = data.split()
	
	print("Got wordsAndTags")

	for wordAndTag in wordsAndTags:
		wordOnly, tagOnly = wordAndTag.split('|')
		strOfWords = strOfWords + wordOnly + " "
		wordOnly = wordOnly.lower()  # Convert to lowercase
		#print strOfWords

numOfUniqueWords = len(Counter(strOfWords.split()))

print ('Num of unique words: %d' % (numOfUniqueWords))

# Read the file for the 2nd time
with open(filename, encoding='utf-8') as f:
	for line in f:

		previous = "<s>"
		endOfLine = "</s>"
		context[previous] += 1
		context[endOfLine] += 1

		# split line into wordtags with " "
		wordtags = line.split()
        
		for wordtag in wordtags:

			#if wordtag not in listOfWords:
			#	listOfWords.append(wordtag)

			# split wordtag into word, tag with "|"
			word, tag = wordtag.split('|')
			word = word.lower()  # Convert to lowercase

			#print 'word, tag'
			#print word
			#print tag
			if tag not in listOfTags:
				listOfTags.append(tag)

			# Count the transition
			transitionKey = previous + " " + tag
			if transitionKey in transition:
				transition[transitionKey] += 1
			else:
				transition[transitionKey] = 1

			# Count the context
			contextKey = tag
			if contextKey in context:
				context[tag] += 1                 
			else:
				context[tag] = 1

			# Count the emission
			emitKey = tag + " " + word
			if emitKey in emit:
				emit[emitKey] += 1
			else:
				emit[emitKey] = 1
			
			previous = tag

		transitionKey = previous + " </s>"
		if transitionKey in transition:
			transition[transitionKey] += 1
		else:
			transition[transitionKey] = 1

		'''
		print transition
		print 'end'

		break
		'''

# Print list of tags
# print (listOfTags)

listOfTags = ["<s>", ",", ".", "–", "-", ":", "CD", "CC", "CP", "DT", "FW", "IN", "JJ", "MD", "NEG", "NN", "OP", "PR", "RB", "RP", "SC", "SYM", "UH", "VB", "WDT", """, """, "\"", "</s>"]

with open('probs_hmm_sunda.txt', 'w', encoding='utf-8') as f:
    # Print the transition (POS -> POS) probabilities
    for key, value in transition.items():
        
        # split key into previous, tag with " "
        previousKey, tagKey = key.split()

        transitionInfo = 'T' + ' ' + key + ' ' + str(value / context[previousKey]) + '\n'
        f.write(transitionInfo)

    # Print the emission (POS -> Word) probabilities with "E"
    for key, value in emit.items():

        # split key into tag, word with " "
        tagEmit, wordEmit = key.split()

        # print the probability
        #print ("E %s %f" % (key, value / context[tag]))

        # Use Laplace Smoothing
        vocabSize = numOfUniqueWords

        # number of occurence of value (pair of tagEmit and wordEmit)
        numOfTimesSymWEmittedAtS = value

        # number of words having tagEmit as the TAG
        totalNumOfSymEmittedByS = context[tagEmit]

        # laplace smoothing probability
        laplaceSmoothingProb = (numOfTimesSymWEmittedAtS + 1) / (totalNumOfSymEmittedByS + vocabSize)

        emitInfo = 'E' + ' ' + key + ' ' + str(laplaceSmoothingProb) + '\n'
            
        f.write(emitInfo)

f.close()


# FINDING POS TAGS


# ===================================================================================================
# =================================== FINDING POS TAG INPUT (HMM) ===================================
# ===================================================================================================
# Membaca input kalimat dari file 'data_input2.txt'
# print("\nInput kalimat: ")
# userInput = input()
print("\n\n")
print("Input kalimat: ")
with open("data/data_input5.txt", "r", encoding="utf-8") as f:
    userInput = f.read()

# Pisahkan kata terakhir jika langsung diikuti oleh tanda titik tanpa spasi
# userInput = re.sub(r'(\w)(\.)$', r'\1 \2', userInput)

# Tokenize input
# tokens = userInput.split()

# Memisahkan kata dan tanda baca, serta membersihkan spasi dan mengubah huruf menjadi lowercase
words = split_punctuation(userInput)
if not userInput:
    print("File input kosong atau tidak valid.")
    sys.exit(1)
words = [word.lower().strip() for word in words]

# Cetak hasil tokenized
print("\nTokenized input:", words)

# Model Loading

# make a map for transition, emission, possible_tags
possible_tags = dict()
transition = dict()
emit = dict()

# Read the file
modelFile = 'probs_hmm_sunda.txt'
with open(modelFile, encoding='utf-8') as f:
	for line in f:
		# split line into type, n_context, word, prob
		typeOfProb, n_context, word, prob = line.split()

		# enumerate all tags
		#possible_tags[n_context] = 1  

		if typeOfProb == 'T':
			transitionInfo = n_context + " " + word
			transition[transitionInfo] = float(prob)
		else:
			emitInfo = n_context + " " + word.lower()
			emit[emitInfo] = float(prob)


# Forward Step

# Analisis kata sebelum Viterbi
print("\nAnalisis Kata Sebelum Viterbi:")
print(f"{'No.':<4} {'Word':<15} {'Method':<10} {'In Emission':<15}")
print("-" * 45)

hmm_words = []
smoothing_words = []

for i, word in enumerate(words, 1):
    # Cek apakah kata ada di emission probability
    in_emission = any(f"{tag} {word}" in emit for tag in listOfTags)
    
    if in_emission:
        method = "HMM"
        hmm_words.append(word)
    else:
        method = "Smoothing"
        smoothing_words.append(word)
        
    print(f"{i:<4} {word:<15} {method:<10} {'Yes' if in_emission else 'No':<15}")

print("\nSummary Sebelum Viterbi:")
print(f"Total words: {len(words)}")
print(f"HMM words: {len(hmm_words)} ({', '.join(hmm_words)})")
print(f"Smoothing words: {len(smoothing_words)} ({', '.join(smoothing_words)})")


# split line into words
words = split_punctuation(userInput)
words = [word.lower().strip() for word in words]

wordsLen = len(words)

# make maps best_score, best_edge
best_score = dict()
best_edge = dict()


# make maps transition, emission
transition_score = dict()
emission_score = dict()


# start with <s>
best_score["0 <s>"] = 0
best_edge["0 <s>"] = None


for i in range(0, wordsLen):
	for prev in listOfTags:
		for next in listOfTags:
			best_score_key = str(i) + " " + prev
			transition_key = prev + " " + next
			emit_key = next + " " + words[i]		
			
			if best_score_key in best_score and transition_key in transition:

				if emit_key not in emit:

					# Use Laplace Smoothing

					# compute emission probability using Laplace Smoothing
					vocabSize2 = numOfUniqueWords

					# number of occurence of value (pair of tagEmit and wordEmit)
					numOfTimesSymWEmittedAtS2 = 0

					# number of words having tagEmit as the TAG
					totalNumOfSymEmittedByS2 = context[next]

					# laplace smoothing probability
					emit[emit_key] = (numOfTimesSymWEmittedAtS2 + 1) / (totalNumOfSymEmittedByS2 + vocabSize2)

				
				score = best_score[best_score_key] - math.log(transition[transition_key]) - math.log(emit[emit_key])

				best_score_key = str(i+1) + " " + next
				best_edge_key = str(i+1) + " " + next

				transProb_key = str(i+1) + " " + next
				emitProb_key = str(i+1) + " " + next

				if best_score_key not in best_score:
					# print ('best_score_key not in best_score')
					best_score[best_score_key] = score
					best_edge[best_edge_key] = str(i) + " " + prev

					transition_score[transProb_key] = transition[transition_key];
					emission_score[emitProb_key] = emit[emit_key];

				elif best_score_key in best_score and best_score[best_score_key] > score:
					# print ('best_score > score')
					best_score[best_score_key] = score
					best_edge[best_edge_key] = str(i) + " " + prev

					transition_score[transProb_key] = transition[transition_key];
					emission_score[emitProb_key] = emit[emit_key];

listOfTagsBeforeEnd = []
listOfProbTagsBeforeEnd = []

for tag in listOfTags:
	best_score_key = str(wordsLen) + " " + tag
	transition_key = tag + " " + "</s>"
	
	if best_score_key in best_score and transition_key in transition:
		listOfTagsBeforeEnd.append(tag)
		listOfProbTagsBeforeEnd.append(best_score[best_score_key] - math.log(transition[transition_key]))


best_score_key = str(wordsLen + 1) + " " + "</s>"
best_edge_key = str(wordsLen + 1) + " " + "</s>"
best_score[best_score_key] = min(listOfProbTagsBeforeEnd)
best_edge[best_edge_key] = str(wordsLen) + " " + listOfTagsBeforeEnd[listOfProbTagsBeforeEnd.index(min(listOfProbTagsBeforeEnd))]



# BACKWARD STEP
tags = []
bestscores = []
transitions = []
emissions = []


#best_edge_key = str(wordsLen + 1) + " " + "</s>"
best_edge_key = str(wordsLen+1) + " " + "</s>"
next_edge = best_edge[best_edge_key]

# append the last best_score to bestscores
bestscores.append(best_score[best_edge_key]);

while next_edge != "0 <s>":
	# Add the substring for this edge to the words
	
	# split next_edge into position, tag
	position, tag = next_edge.split()

	# append tag to tags
	tags.append(tag)

	# append bestscore to bestscores
	bestscores.append(best_score[next_edge]);

	# append transition and emission to transitions and emissions
	transitions.append(transition_score[next_edge]);
	emissions.append(emission_score[next_edge]);

	next_edge = best_edge[next_edge]

tags.reverse()
bestscores.reverse()
transitions.reverse()
emissions.reverse()


# join tags into a string and print
print("\n")

# print("Sequence of tags")
# print(tags)

print("\n")

# print("Best score of each word")
# print(bestscores)
# print("\n")


print("\nTotal of best score (log-prob):")
total_log_score = sum(bestscores)
print(total_log_score)

print("\nTotal of transition probabilities (log-prob):")
total_log_trans = sum(math.log(vl) for vl in transitions if vl > 0)
print(total_log_trans)

print("\nTotal of emission probabilities (log-prob):")
total_log_emit = sum(math.log(vl) for vl in emissions if vl > 0)
print(total_log_emit)

# print("Total of best score")
# totalVal = 1
# for vl in bestscores:
# 	totalVal = totalVal * vl
# print(totalVal)
# print("\n")

# print("Transition probabilities")
# print(transitions)
# print("\n")
# print("Total of transition probabilities")
# totalVal = 1
# for vl in transitions:
# 	totalVal = totalVal * vl
# print(totalVal)
# print("\n")

# print("Emission probabilities")
# print(emissions)
# print("\n")
# print("Total of emission probabilities")
# totalVal = 1
# for vl in emissions:
# 	totalVal = totalVal * vl
# print (totalVal)


# ===================================================================================================
# =================================== EVALUATION  
# ===================================================================================================
def read_and_extract_labels(file_path):
    # Membaca file dan mengekstrak kelas label dari setiap kata
    y_true = []
    
    # Coba membuka file dengan encoding 'utf-8'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Menghapus spasi dan baris kosong
                line = line.strip()

                # Memisahkan kata dan label menggunakan karakter '|'
                words_labels = line.split()

                for word_label in words_labels:
                    word, label = word_label.split('|')
                    y_true.append(label)
    except UnicodeDecodeError:
        print("Gagal membuka file dengan encoding 'utf-8'. Coba gunakan encoding lain.")
    
    return y_true

# Path ke file teks yang akan dibaca
file_path = 'data/korpus_validasi5.txt'

# Mendapatkan hasil output
y_true = read_and_extract_labels(file_path)

# Menampilkan hasil
y_pred=tags

def calculate_metrics_macro(y_true, y_pred, beta=0.5):
    if not y_true or len(y_true) != len(y_pred):
        print(f"Peringatan: Validasi tidak dapat dilakukan karena data tidak sesuai.")
        return None

    label_set = set(y_true) | set(y_pred)
    per_label = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            per_label[yt]['tp'] += 1
        else:
            per_label[yp]['fp'] += 1
            per_label[yt]['fn'] += 1

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    f_beta_sum = 0
    n_labels = len(label_set)

    for label in label_set:
        tp = per_label[label]['tp']
        fp = per_label[label]['fp']
        fn = per_label[label]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f_beta = (
            (1 + beta ** 2) * precision * recall /
            ((beta ** 2 * precision) + recall)
        ) if (precision + recall) > 0 else 0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        f_beta_sum += f_beta

    macro_precision = precision_sum / n_labels
    macro_recall = recall_sum / n_labels
    macro_f1 = f1_sum / n_labels
    macro_f_beta = f_beta_sum / n_labels
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        f"f{beta}_score": macro_f_beta
    }

beta = 0.5  # Ganti dengan 2 jika ingin menggunakan F2-Score
    
if y_true:
    metrics = calculate_metrics_macro(y_true, tags, beta=beta)
    if metrics:
        print("\n================== HASIL EVALUASI HMM ==================")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"F{beta}-Score: {metrics[f'f{beta}_score']:.4f}")




# Hitung jumlah kata yang tertangani oleh HMM (in-vocab) dan smoothing (OOV)
hmm_count = 0
smoothing_count = 0

for word in words:
    if word in context:  # context berisi semua tag, bukan kata!
        # Perlu vocabulary, bukan context
        pass

# Buat vocabulary dari data train
vocabulary = set(strOfWords.split())

for word in words:
    if word in vocabulary:
        hmm_count += 1
    else:
        smoothing_count += 1


print(f"Jumlah kata yang tertangani oleh HMM (in-vocab): {hmm_count}")
print(f"Jumlah kata yang tertangani oleh smoothing (OOV): {smoothing_count}")
# Simpan hasil prediksi yang salah ke file
wrong_file = "hasil/hmm/Hybrid_HMM_First_Wrong.txt"
with open(wrong_file, 'w', encoding='utf-8') as f:
    f.write("No.\tKata\tPrediksi\tLabel_Benar\n")
    f.write("-" * 40 + "\n")
    for i, (word, pred, gold) in enumerate(zip(words, tags, y_true)):
        if pred != gold:
            f.write(f"{i+1}\t{word}\t{pred}\t{gold}\n")
print(f"Hasil prediksi yang salah telah disimpan di {wrong_file}")

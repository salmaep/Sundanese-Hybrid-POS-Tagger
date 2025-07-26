import re
import json
import os
from rule_based5 import RuleBased
from collections import defaultdict
import sys
from io import StringIO

class TeeStringIO(StringIO):
    def __init__(self, filename, mode='w', encoding='utf-8'):
        super().__init__()
        self.file = open(filename, mode, encoding=encoding)
        self.original_stdout = sys.stdout

    def write(self, s):
        self.original_stdout.write(s)
        self.file.write(s)
        super().write(s)

    def close(self):
        self.file.close()
        super().close()

# Fungsi untuk tokenisasi
def split_punctuation(text):
    pattern = r"""
        (?:
            \b(?:HR|QS|M|E|R|S|H)\.                          # HR. atau QS.
            | [A-Z][a-z]{1,3}\.                      # Prof., Dr., Mr., dll
            | [A-Za-z]+(?:\.[A-Za-z]+)+              # QS.A.S., A.S., dkk
            | \b[A-Z]{2,5}\$                         # Gabungan simbol mata uang, contoh: US$
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

# Fungsi untuk menghitung metrik evaluasi
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

def calculate_per_tag_metrics(y_true, y_pred):
    if not y_true or len(y_true) != len(y_pred):
        return None
    
    # Initialize counters for each tag
    tag_metrics = defaultdict(lambda: {'correct': 0, 'predicted': 0, 'actual': 0})
    
    # Count occurrences
    for yt, yp in zip(y_true, y_pred):
        tag_metrics[yt]['actual'] += 1
        tag_metrics[yp]['predicted'] += 1
        if yt == yp:
            tag_metrics[yt]['correct'] += 1
    
    # Calculate metrics for each tag
    results = {}
    for tag, counts in tag_metrics.items():
        precision = counts['correct'] / counts['predicted'] if counts['predicted'] > 0 else 0
        recall = counts['correct'] / counts['actual'] if counts['actual'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[tag] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': counts['actual']
        }
    
    return results

# Fungsi untuk memeriksa validitas tag yang dihasilkan dari aturan affix
def validate_affix_tags(affix_word_tags, validation_file):
    correct_count = 0
    incorrect_count = 0
    incorrect_details = []

    # Membaca file validasi
    with open(validation_file, 'r', encoding='utf-8') as val_file:
        validation_data = {}
        for line in val_file:
            line = line.strip()
            if not line:
                continue
            tagged_tokens = line.split()
            for token_label in tagged_tokens:
                if '|' not in token_label:
                    continue
                word, label = token_label.split('|')
                validation_data[word.lower()] = label  # Simpan label dengan kata dalam huruf kecil

    # Memeriksa setiap kata yang ditandai dengan aturan affix
    for word, tag in affix_word_tags:
        correct_tag = validation_data.get(word.lower())
        if correct_tag:
            if tag == correct_tag:
                correct_count += 1
            else:
                incorrect_count += 1
                incorrect_details.append((word, tag, correct_tag))
        else:
            # Jika kata tidak ditemukan di data validasi
            incorrect_count += 1
            incorrect_details.append((word, tag, "Tidak Ada"))

    return correct_count, incorrect_count, incorrect_details

# def process_file(input_file, output_file, validation_file):
#     log_file = "hasil/rule_based/log_hasil.txt"
#     output_capture = TeeStringIO(log_file)
#     sys.stdout = output_capture

#     rule_based = RuleBased()

#     total_tokens = 0
#     unk_count = 0
#     y_true = []
#     y_pred = []
#     unknown_words = defaultdict(int)
#     word_tag_pairs = []
#     affix_word_tags = []
#     validation_data = {}

#     output_tagged_file = "hasil/rule_based/output_rule_based_data3.txt"
#     affix_analysis_file = "hasil/rule_based/analisis_rule_affix_data3.txt"

#     with open(input_file, 'r', encoding='utf-8') as infile, \
#          open(output_file, 'w', encoding='utf-8') as outfile, \
#          open(output_tagged_file, 'w', encoding='utf-8') as tagged_outfile:

#         for line in infile:
#             tokens = split_punctuation(line.strip())
#             tagged_tokens = []

#             for idx, token in enumerate(tokens):
#                 total_tokens += 1
#                 prev_token = tokens[idx - 1] if idx > 0 else "UNK"
#                 next_token = tokens[idx + 1] if idx < len(tokens) - 1 else "UNK"
#                 prev_tag = y_pred[idx - 1] if idx > 0 else "<s>"

#                 result = rule_based.check_lexicon_and_rules(token.lower(), prev_token, next_token, prev_tag=prev_tag)
#                 if isinstance(result, tuple):
#                     _, tag = result
#                 else:
#                     tag = result

#                 if tag != "UNK":
#                     affix_word_tags.append((token.lower(), tag))
#                 else:
#                     unk_count += 1
#                     unknown_words[token.lower()] += 1

#                 tagged_tokens.append(f"{token}|{tag}")
#                 y_pred.append(tag)
#                 word_tag_pairs.append((token, tag))

#             tagged_outfile.write(" ".join(tagged_tokens) + "\n")
#             outfile.write(" ".join(tagged_tokens) + "\n")

#     # Load validasi
#     with open(validation_file, 'r', encoding='utf-8') as val_file:
#         for line in val_file:
#             for token_label in line.strip().split():
#                 if "|" in token_label:
#                     word, label = token_label.split("|")
#                     y_true.append(label)
#                     validation_data[word.lower()] = label

#     min_len = min(len(y_true), len(y_pred))
#     y_true = y_true[:min_len]
#     y_pred = y_pred[:min_len]

#     # Analisis affix
#     with open(affix_analysis_file, 'w', encoding='utf-8') as f:
#         f.write("Analisis Penggunaan Rule Affix\n")
#         correct_count = 0
#         incorrect_count = 0

#         for rule_id, word_rule_pairs in sorted(rule_based.affix_rule_usage.items()):
#             f.write(f"\nRule #{rule_id}:\n")
#             for word, rule_key in word_rule_pairs:
#                 predicted_tag = next((tag for w, tag in affix_word_tags if w == word), "UNK")
#                 correct_tag = validation_data.get(word, "Tidak Ada")
#                 status = "Benar" if predicted_tag == correct_tag else "Salah"
#                 if status == "Benar":
#                     correct_count += 1
#                 else:
#                     incorrect_count += 1

#                 f.write(f"Kata: {word}, Rule: {rule_key}, Prediksi: {predicted_tag}, Validasi: {correct_tag}, Status: {status}\n")

#         f.write(f"\nJumlah Benar: {correct_count}\n")
#         f.write(f"Jumlah Salah: {incorrect_count}\n")

#     # Evaluasi
#     metrics = calculate_metrics_macro(y_true, y_pred, beta=0.5)
#     tag_metrics = calculate_per_tag_metrics(y_true, y_pred)

#     if metrics:
#         print("\n================== HASIL EVALUASI KESELURUHAN ==================")
#         print(f"Accuracy: {metrics['accuracy']:.4f}")
#         print(f"Precision: {metrics['precision']:.4f}")
#         print(f"Recall: {metrics['recall']:.4f}")
#         print(f"F1-Score: {metrics['f1_score']:.4f}")
#         print(f"F0.5-Score: {metrics['f0.5_score']:.4f}")

#     sys.stdout = output_capture.original_stdout
#     output_capture.close()

def process_file_generic(rule_based, input_file, output_file, validation_file,
                         log_file, tagged_out_file, affix_analysis_file):
    output_capture = TeeStringIO(log_file)
    sys.stdout = output_capture

    total_tokens = 0
    unk_count = 0
    y_true = []
    y_pred = []
    unknown_words = defaultdict(int)
    word_tag_pairs = []
    affix_word_tags = []
    validation_data = {}

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(tagged_out_file, 'w', encoding='utf-8') as tagged_outfile:

        for line in infile:
            tokens = split_punctuation(line.strip())
            tagged_tokens = []

            for idx, token in enumerate(tokens):
                total_tokens += 1
                prev_token = tokens[idx - 1] if idx > 0 else "UNK"
                next_token = tokens[idx + 1] if idx < len(tokens) - 1 else "UNK"
                prev_tag = y_pred[idx - 1] if idx > 0 else "<s>"

                result = rule_based.check_lexicon_and_rules(token.lower(), prev_token, next_token, prev_tag=prev_tag)
                if isinstance(result, tuple):
                    _, tag = result
                else:
                    tag = result

                if tag != "UNK":
                    affix_word_tags.append((token.lower(), tag))
                else:
                    unk_count += 1
                    unknown_words[token.lower()] += 1

                tagged_tokens.append(f"{token}|{tag}")
                y_pred.append(tag)
                word_tag_pairs.append((token, tag))

            tagged_outfile.write(" ".join(tagged_tokens) + "\n")
            outfile.write(" ".join(tagged_tokens) + "\n")

    # Load validasi
    with open(validation_file, 'r', encoding='utf-8') as val_file:
        for line in val_file:
            for token_label in line.strip().split():
                if "|" in token_label:
                    word, label = token_label.split("|")
                    y_true.append(label)
                    validation_data[word.lower()] = label

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Analisis affix
    with open(affix_analysis_file, 'w', encoding='utf-8') as f:
        f.write("Analisis Penggunaan Rule Affix\n")
        correct_count = 0
        incorrect_count = 0

        for rule_id, word_rule_pairs in sorted(rule_based.affix_rule_usage.items()):
            f.write(f"\nRule #{rule_id}:\n")
            for word, rule_key in word_rule_pairs:
                predicted_tag = next((tag for w, tag in affix_word_tags if w == word), "UNK")
                correct_tag = validation_data.get(word, "Tidak Ada")
                status = "Benar" if predicted_tag == correct_tag else "Salah"
                if status == "Benar":
                    correct_count += 1
                else:
                    incorrect_count += 1
                f.write(f"Kata: {word}, Rule: {rule_key}, Prediksi: {predicted_tag}, Validasi: {correct_tag}, Status: {status}\n")

        f.write(f"\nJumlah Benar: {correct_count}\n")
        f.write(f"Jumlah Salah: {incorrect_count}\n")

    # Evaluasi metrik keseluruhan
    metrics = calculate_metrics_macro(y_true, y_pred, beta=0.5)
    tag_metrics = calculate_per_tag_metrics(y_true, y_pred)

    if metrics:
        print("\n================== HASIL EVALUASI KESELURUHAN ==================")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"F0.5-Score: {metrics['f0.5_score']:.4f}")

        if tag_metrics:
            print("\n================== HASIL EVALUASI PER TAG ==================")
            print(f"{'TAG':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print("-" * 55)
            for tag, m in sorted(tag_metrics.items(), key=lambda x: x[1]['support'], reverse=True):
                print(f"{tag:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")

    sys.stdout = output_capture.original_stdout
    output_capture.close()

    return {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1_score"],
        "F0.5-Score": metrics["f0.5_score"]
    }

def process_file_with_rule(rule_based, input_file, output_file, validation_file):
    log_file = output_file.replace(".txt", "_log.txt")
    tagged_out_file = output_file.replace(".txt", "_tagged.txt")
    affix_analysis_file = output_file.replace(".txt", "_affix_analysis.txt")

    return process_file_generic(
        rule_based,
        input_file,
        output_file,
        validation_file,
        log_file,
        tagged_out_file,
        affix_analysis_file
    )


# Ganti 'data/data_input_luar_backup.txt', 'data/output.txt', dan 'data/validation.txt' dengan nama file yang sesuai
# process_file('data/data_input_luar_backup.txt', 'hasil/hasil_rb_data_uji_luar_eksperimen.txt', 'data/data_uji_luar_validasi.txt')
# process_file('data/data_input5_baru.txt', 'hasil/rule_based/hasil_rb_data_luar.txt', 'data/korpus_validasi5.txt')
# process_file('data/data_input22.txt', 'hasil/rule_based/hasil_rb_data2_5k.txt', 'data/korpus_validasi_ID_5K.txt')
# process_file('data/data_kalimat.txt', 'hasil/hasil_kalimat.txt', 'data/data_kalimat_validasi.txt')
# process_file('data/data_input5.txt', 'hasil/rule_based/hasil_rb_data_ood.txt', 'data/korpus_validasi5.txt')

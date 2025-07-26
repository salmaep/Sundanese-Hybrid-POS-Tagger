from __future__ import division
from collections import Counter
from collections import defaultdict
import math
import re
import sys
from io import StringIO
from rule_based import RuleBased  
import os
from matplotlib.cbook import flatten

# ===================================================================================================
#                                     UTILITY CLASSES
# ===================================================================================================

class TeeStringIO(StringIO):
    """Kelas untuk mengalihkan output ke string dan file sekaligus"""
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

# ===================================================================================================
#                                     TOKENIZER CLASS
# ===================================================================================================

class Tokenizer:
    """Class untuk tokenisasi teks"""
    
    @staticmethod
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


# ===================================================================================================
#                                     EVALUATION CLASS
# ===================================================================================================

class Evaluator:
    """Class untuk evaluasi dan perhitungan metrik"""
    
    @staticmethod
    def read_and_extract_labels(file_path):
        """Membaca file dan mengekstrak label dari format word|TAG"""
        y_true = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    words_labels = line.split()
                    for word_label in words_labels:
                        word, label = word_label.split('|')
                        y_true.append(label)
        except UnicodeDecodeError:
            print("Gagal membuka file dengan encoding 'utf-8'. Coba gunakan encoding lain.")
        return y_true

    @staticmethod
    def calculate_metrics_macro(y_true, y_pred, beta=0.5):
        """Menghitung metrik evaluasi macro-averaged"""
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

    @staticmethod
    def calculate_tag_metrics_rulebased(y_true, y_pred):
        """Menghitung metrik per tag untuk rule-based"""
        tag_metrics = {}
        tags_set = set(y_true) | set(y_pred)
        for tag in tags_set:
            tp = sum((yt == tag and yp == tag) for yt, yp in zip(y_true, y_pred))
            fp = sum((yt != tag and yp == tag) for yt, yp in zip(y_true, y_pred))
            fn = sum((yt == tag and yp != tag) for yt, yp in zip(y_true, y_pred))
            support = sum(yt == tag for yt in y_true)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            tag_metrics[tag] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        return tag_metrics

    @staticmethod
    def calculate_macro_metrics_rulebased(y_true, y_pred, beta=0.5):
        """Menghitung metrik macro untuk rule-based"""
        tag_metrics = Evaluator.calculate_tag_metrics_rulebased(y_true, y_pred)
        n = len(tag_metrics)
        macro_precision = sum(m['precision'] for m in tag_metrics.values()) / n if n else 0
        macro_recall = sum(m['recall'] for m in tag_metrics.values()) / n if n else 0
        macro_f1 = sum(m['f1'] for m in tag_metrics.values()) / n if n else 0
        macro_fbeta = 0
        for m in tag_metrics.values():
            p, r = m['precision'], m['recall']
            if p + r > 0:
                macro_fbeta += (1 + beta**2) * p * r / (beta**2 * p + r)
        macro_fbeta = macro_fbeta / n if n else 0
        accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) if y_true else 0
        return {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            f"f{beta}_score": macro_fbeta,
            "tag_metrics": tag_metrics
        }

# ===================================================================================================
#                                     HMM TRAINER CLASS
# ===================================================================================================

class HMMTrainer:
    """Class untuk training model HMM"""
    
    def __init__(self):
        self.context = dict()
        self.emit = dict()
        self.transition = dict()
        self.vocabulary = set()
        self.num_unique_words = 0
        self.list_of_tags = ["<s>", ",", ".", "–", "-", ":", "CD", "CC", "CP", "DT", "FW", "IN", "JJ", "MD", "NEG", "NN", "OP", "PR", "RB", "RP", "SC", "SYM", "UH", "VB", "WDT", """, """, "\"","X", "</s>"]
    
    def train(self, filename):
        """Training model HMM dari file training data"""
        print('Start Training HMM')
        
        # Initialize
        previous = "<s>"
        endOfLine = "</s>"
        self.context[previous] = 0
        self.context[endOfLine] = 0
        str_of_words = ""

        # Read file pertama kali untuk menghitung vocabulary
        with open(filename, encoding='utf-8') as f:
            data = f.read().replace('\n', ' ')
            words_and_tags = data.split()
            print("Got wordsAndTags")

            for word_and_tag in words_and_tags:
                word_only, tag_only = word_and_tag.split('|')
                word_only = word_only.lower()
                str_of_words = str_of_words + word_only + " "

        self.num_unique_words = len(Counter(str_of_words.split()))
        print('Num of unique words: %d' % (self.num_unique_words))
        
        # Create vocabulary set untuk OOV detection
        self.vocabulary = set(str_of_words.split())

        # Read file kedua kali untuk training
        with open(filename, encoding='utf-8') as f:
            for line in f:
                previous = "<s>"
                endOfLine = "</s>"
                self.context[previous] += 1
                self.context[endOfLine] += 1

                wordtags = line.split()
                
                for wordtag in wordtags:
                    word, tag = wordtag.split('|')
                    word = word.lower()

                    # Count the transition
                    transition_key = previous + " " + tag
                    if transition_key in self.transition:
                        self.transition[transition_key] += 1
                    else:
                        self.transition[transition_key] = 1

                    # Count the context
                    context_key = tag
                    if context_key in self.context:
                        self.context[tag] += 1                 
                    else:
                        self.context[tag] = 1

                    # Count the emission
                    emit_key = tag + " " + word
                    if emit_key in self.emit:
                        self.emit[emit_key] += 1
                    else:
                        self.emit[emit_key] = 1
                    
                    previous = tag

                transition_key = previous + " </s>"
                if transition_key in self.transition:
                    self.transition[transition_key] += 1
                else:
                    self.transition[transition_key] = 1

    def save_model(self, filename='probs_hmm_sunda.txt'):
        """Menyimpan model HMM ke file"""
        with open(filename, 'w', encoding='utf-8') as f:
            # Print the transition (POS -> POS) probabilities
            for key, value in self.transition.items():
                previous_key, tag_key = key.split()
                transition_info = 'T' + ' ' + key + ' ' + str(value / self.context[previous_key]) + '\n'
                f.write(transition_info)

            # Print the emission (POS -> Word) probabilities with Laplace Smoothing
            for key, value in self.emit.items():
                tag_emit, word_emit = key.split()
                vocab_size = self.num_unique_words
                num_of_times_sym_w_emitted_at_s = value
                total_num_of_sym_emitted_by_s = self.context[tag_emit]
                laplace_smoothing_prob = (num_of_times_sym_w_emitted_at_s + 1) / (total_num_of_sym_emitted_by_s + vocab_size)
                emit_info = 'E' + ' ' + key + ' ' + str(laplace_smoothing_prob) + '\n'
                f.write(emit_info)

# ===================================================================================================
#                                     HMM DECODER CLASS
# ===================================================================================================

class HMMDecoder:
    """Class untuk decoding menggunakan model HMM dengan Viterbi algorithm"""
    
    def __init__(self):
        self.transition = dict()
        self.emit = dict()
        self.list_of_tags = ["<s>", ",", ".", "–", "-", ":", "CD", "CC", "CP", "DT", "FW", "IN", "JJ", "MD", "NEG", "NN", "OP", "PR", "RB", "RP", "SC", "SYM", "UH", "VB", "WDT", """, """, "\"","X", "</s>"]
        self.vocabulary = set()
        self.num_unique_words = 0
        self.context = dict()
    
    def load_model(self, model_file='probs_hmm_sunda.txt'):
        """Load model HMM dari file"""
        with open(model_file, encoding='utf-8') as f:
            for line in f:
                type_of_prob, n_context, word, prob = line.split()

                if type_of_prob == 'T':
                    transition_info = n_context + " " + word
                    self.transition[transition_info] = float(prob)
                else:
                    emit_info = n_context + " " + word.lower()
                    self.emit[emit_info] = float(prob)
    
    def set_training_data(self, vocabulary, num_unique_words, context):
        """Set data training yang diperlukan untuk Laplace smoothing"""
        self.vocabulary = vocabulary
        self.num_unique_words = num_unique_words
        self.context = context
    
    def decode(self, words):
        """Decode sequence of words menggunakan Viterbi algorithm"""
        print("\n================== STEP 1: RUNNING PURE HMM ==================")
        
        words_len = len(words)
        best_score = dict()
        best_edge = dict()
        oov_indices = []  # Track OOV words for later processing

        # make maps transition, emission
        transition_score = dict()
        emission_score = dict()

        # Start with <s>
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None

        # Forward Step - Pure HMM with Laplace Smoothing
        for i in range(0, words_len):
            current_word = words[i]
            
            # Check if word is OOV
            is_oov = current_word not in self.vocabulary
            if is_oov:
                oov_indices.append(i)
                print(f"Word '{current_word}' at position {i} is OOV - will be processed with Laplace smoothing")
            
            for prev in self.list_of_tags:
                for next_tag in self.list_of_tags:
                    best_score_key = str(i) + " " + prev
                    transition_key = prev + " " + next_tag
                    emit_key = next_tag + " " + current_word

                    if best_score_key in best_score and transition_key in self.transition:
                        
                        # Handle emission probability - Pure HMM with Laplace Smoothing
                        if emit_key not in self.emit:
                            # Apply Laplace smoothing for all unseen words
                            vocab_size2 = self.num_unique_words
                            num_of_times_sym_w_emitted_at_s2 = 0
                            total_num_of_sym_emitted_by_s2 = self.context[next_tag]
                            self.emit[emit_key] = (num_of_times_sym_w_emitted_at_s2 + 1) / (total_num_of_sym_emitted_by_s2 + vocab_size2)

                        score = best_score[best_score_key] - math.log(self.transition[transition_key]) - math.log(self.emit[emit_key])

                        best_score_key_next = str(i+1) + " " + next_tag
                        best_edge_key = str(i+1) + " " + next_tag

                        trans_prob_key = str(i+1) + " " + next_tag
                        emit_prob_key = str(i+1) + " " + next_tag

                        if best_score_key_next not in best_score:
                            best_score[best_score_key_next] = score
                            best_edge[best_edge_key] = str(i) + " " + prev

                            transition_score[trans_prob_key] = self.transition[transition_key]
                            emission_score[emit_prob_key] = self.emit[emit_key]
                            
                        elif best_score_key_next in best_score and best_score[best_score_key_next] > score:
                            best_score[best_score_key_next] = score
                            best_edge[best_edge_key] = str(i) + " " + prev

                            transition_score[trans_prob_key] = self.transition[transition_key]
                            emission_score[emit_prob_key] = self.emit[emit_key]

        # Find best final state
        list_of_tags_before_end = []
        list_of_prob_tags_before_end = []

        for tag in self.list_of_tags:
            best_score_key = str(words_len) + " " + tag
            transition_key = tag + " " + "</s>"
            
            if best_score_key in best_score and transition_key in self.transition:
                list_of_tags_before_end.append(tag)
                list_of_prob_tags_before_end.append(best_score[best_score_key] - math.log(self.transition[transition_key]))

        best_score_key = str(words_len + 1) + " " + "</s>"
        best_edge_key = str(words_len + 1) + " " + "</s>"
        best_score[best_score_key] = min(list_of_prob_tags_before_end)
        best_edge[best_edge_key] = str(words_len) + " " + list_of_tags_before_end[list_of_prob_tags_before_end.index(min(list_of_prob_tags_before_end))]

        # Backtracking
        print("\n================== STEP 2: BACKTRACKING - GET HMM RESULTS ==================")
        
        hmm_tags = []
        bestscores = []
        transitions = []
        emissions = []

        best_edge_key = str(words_len+1) + " " + "</s>"
        next_edge = best_edge[best_edge_key]

        bestscores.append(best_score[best_edge_key])

        while next_edge != "0 <s>":
            position, tag = next_edge.split()
            hmm_tags.append(tag)
            bestscores.append(best_score[next_edge])
            transitions.append(transition_score[next_edge])
            emissions.append(emission_score[next_edge])
            next_edge = best_edge[next_edge]

        hmm_tags.reverse()
        bestscores.reverse()
        transitions.reverse()
        emissions.reverse()

        # print("HMM Results (before OOV replacement):")
        print("\nSequence of HMM tags:")
        print(hmm_tags)

        return {
            'hmm_tags': hmm_tags,
            'oov_indices': oov_indices,
            'bestscores': bestscores,
            'transitions': transitions,
            'emissions': emissions
        }

# ===================================================================================================
#                                     HYBRID SYSTEM CLASS
# ===================================================================================================

class HybridPOSTagger:
    """Main class untuk hybrid POS tagging system"""
    
    def __init__(self):
        self.rule_based = RuleBased()
        self.hmm_trainer = HMMTrainer()
        self.hmm_decoder = HMMDecoder()
        self.tokenizer = Tokenizer()
        self.evaluator = Evaluator()
    
    def train_hmm(self, training_file):
        """Train HMM model"""
        self.hmm_trainer.train(training_file)
        self.hmm_trainer.save_model()
        
        # Set training data untuk decoder
        self.hmm_decoder.set_training_data(
            self.hmm_trainer.vocabulary,
            self.hmm_trainer.num_unique_words,
            self.hmm_trainer.context
        )
    
    def process_oov_words(self, words, hmm_tags, oov_indices):
        """Process OOV words menggunakan rule-based approach"""
        print("\n================== STEP 3: ANALYZE OOV AND REPLACE WITH RULE-BASED ==================")
        
        # Initialize final results with HMM results
        final_tags = hmm_tags.copy()
        tag_sources = ["hmm"] * len(words)
        words_len = len(words)

        # Statistics tracking
        oov_stats = {
            'total_words': len(words),
            'total_oov': len(oov_indices),
            'numeric_handled': 0,
            'lexicon_handled': 0,
            'affix_rules_handled': 0,
            'nasal_rules_handled': 0,
            'syntax_rules_handled': 0,
            'unknown_handled': 0,
            'hmm_handled': len(words) - len(oov_indices),
            'oov_words': [],
            'hmm_words': []
        }

        # Process each OOV word
        for i in oov_indices:
            current_word = words[i]
            prev_word = words[i-1] if i > 0 else ""
            next_word = words[i+1] if i < words_len-1 else ""
            prev_tag = final_tags[i-1] if i > 0 else "<s>"
            
            print(f"\nProcessing OOV word '{current_word}' at position {i}")
            print(f"HMM assigned tag: {hmm_tags[i]}")
            print(f"Context: prev='{prev_word}', next='{next_word}', prev_tag='{prev_tag}'")
            
            # Apply rule-based approach
            root_word, rule_based_tag = self.rule_based.check_lexicon_and_rules(
                current_word, prev_word, next_word, prev_tag
            )
            
            method_used = self.rule_based.last_method_used
            rule_applied = self.rule_based.last_applied_rule
            
            print(f"Rule-based result: {root_word} -> {rule_based_tag} (method: {method_used})")
            
            # Replace HMM tag with rule-based tag
            if rule_based_tag != "UNK":
                final_tags[i] = rule_based_tag
                
                # Update statistics
                if method_used == "numeric":
                    oov_stats['numeric_handled'] += 1
                    tag_sources[i] = "numeric"
                    status = "Numeric"
                elif method_used == "lexicon":
                    oov_stats['lexicon_handled'] += 1
                    tag_sources[i] = "lexicon"
                    status = "Lexicon"
                elif method_used == "affix_rules":
                    oov_stats['affix_rules_handled'] += 1
                    tag_sources[i] = "affix_rules"
                    status = "Affix Rules"
                elif method_used == "nasal_rules":
                    oov_stats['nasal_rules_handled'] += 1
                    tag_sources[i] = "nasal_rules"
                    status = "Nasal Rules"
                elif method_used == "syntax_rules":
                    oov_stats['syntax_rules_handled'] += 1
                    tag_sources[i] = "syntax_rules"
                    status = "Syntax Rules"
                else:
                    status = "Rule-based (unknown)"
                    tag_sources[i] = "rule_based"
                
                print(f"Tag replaced: {hmm_tags[i]} -> {rule_based_tag} (method: {status})")
                oov_stats['oov_words'].append((i, current_word, rule_based_tag, status))
                
            else:
                # Default to NN instead of keeping HMM tag
                oov_stats['unknown_handled'] += 1
                tag_sources[i] = "default_nn"
                final_tags[i] = "NN"  # Set tag to NN
                print(f"Rule-based returned UNK, setting default tag: NN (was {hmm_tags[i]})")
                oov_stats['oov_words'].append((i, current_word, "NN", "Default NN"))

        # Track HMM words
        for i in range(len(words)):
            if i not in oov_indices:
                oov_stats['hmm_words'].append((i, words[i]))

        return final_tags, tag_sources, oov_stats
    
    def tag_sentence(self, input_text):
        """Tag kalimat input dengan hybrid system"""
        # Tokenize input
        words = self.tokenizer.split_punctuation(input_text)
        words = [word.lower().strip() for word in words]
        
        print("\nTokenized input:", words)
        
        # Load HMM model
        self.hmm_decoder.load_model()
        
        # Run HMM decoding
        hmm_results = self.hmm_decoder.decode(words)
        
        # Process OOV words with rule-based
        final_tags, tag_sources, oov_stats = self.process_oov_words(
            words, 
            hmm_results['hmm_tags'], 
            hmm_results['oov_indices']
        )
        
        return {
            'words': words,
            'final_tags': final_tags,
            'hmm_tags': hmm_results['hmm_tags'],
            'tag_sources': tag_sources,
            'oov_stats': oov_stats,
            'bestscores': hmm_results['bestscores'],
            'transitions': hmm_results['transitions'],
            'emissions': hmm_results['emissions']
        }
    
    def evaluate_and_save_results(self, results, validation_file, output_dir):
        """Evaluasi hasil dan simpan ke file"""
        words = results['words']
        final_tags = results['final_tags']
        hmm_tags = results['hmm_tags']
        tag_sources = results['tag_sources']
        oov_stats = results['oov_stats']
        bestscores = results['bestscores']
        transitions = results['transitions']
        emissions = results['emissions']
        
        # Print final results
        print("\n================== FINAL HYBRID RESULTS ==================")
        print("Final tagged sequence:")
        for i, (word, tag, source) in enumerate(zip(words, final_tags, tag_sources)):
            print(f"{i+1}. {word} -> {tag} (source: {source})")

        print("\nSequence of final tags:")
        print(final_tags)

        print(f"\nTotal of best score (log-prob): {sum(bestscores)}")
        print(f"Total of transition probabilities (log-prob): {sum(math.log(vl) for vl in transitions if vl > 0)}")
        print(f"Total of emission probabilities (log-prob): {sum(math.log(vl) for vl in emissions if vl > 0)}")

        # Evaluation
        y_true = self.evaluator.read_and_extract_labels(validation_file)
        y_pred = final_tags
        beta = 0.5
            
        if y_true:
            metrics = self.evaluator.calculate_metrics_macro(y_true, y_pred, beta=beta)
            if metrics:
                print("\n================== HASIL EVALUASI HYBRID SYSTEM ==================")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-Score: {metrics['f1_score']:.4f}")
                print(f"F{beta}-Score: {metrics[f'f{beta}_score']:.4f}")

        # Statistics summary
        total_token = len(words)
        hmm_count = tag_sources.count("hmm")
        numeric_count = tag_sources.count("numeric")
        lexicon_count = tag_sources.count("lexicon")
        affix_rules_count = tag_sources.count("affix_rules")
        nasal_rules_count = tag_sources.count("nasal_rules")
        syntax_rules_count = tag_sources.count("syntax_rules")
        default_nn = tag_sources.count("default_nn")

        print("\n================== RINGKASAN PENANGANAN TOKEN ==================")
        print(f"Jumlah token: {total_token}")
        print(f"Jumlah token OOV: {len(oov_stats['oov_words'])}")
        print(f"Jumlah token yang tertangani HMM: {hmm_count}")
        print(f"Jumlah token yang tertangani numeric rules: {numeric_count}")
        print(f"Jumlah token yang tertangani lexicon: {lexicon_count}")
        print(f"Jumlah token yang tertangani affix rules: {affix_rules_count}")
        print(f"Jumlah token yang tertangani nasal rules: {nasal_rules_count}")
        print(f"Jumlah token yang tertangani syntax rules: {syntax_rules_count}")
        print(f"Jumlah token OOV yang tetap menggunakan default NN: {default_nn}")

        # PERBAIKAN: Analisis rule-based yang lebih detail
        rb_indices = [i for i, src in enumerate(tag_sources) if src != "hmm"]
        rb_tags = [final_tags[i] for i in rb_indices]
        rb_true = [y_true[i] for i in rb_indices] if y_true else []

        # Hitung jumlah kata per jenis tag hasil rule-based
        rb_tag_count = Counter(rb_tags)

        # Hitung benar/salah per tag
        rb_correct_count = Counter()
        if rb_true:
            for pred, gold in zip(rb_tags, rb_true):
                if pred == gold:
                    rb_correct_count[pred] += 1

        # Tampilkan ringkasan tag hasil rule-based
        if rb_tags and rb_true:
            beta = 0.5
            rb_metrics = self.evaluator.calculate_macro_metrics_rulebased(rb_true, rb_tags, beta=beta)
            tag_metrics = rb_metrics["tag_metrics"]

            print("\n================== ANALISIS KHUSUS RULE-BASED ==================")
            print(f"Total kata yang di-tag rule-based: {len(rb_tags)}")
            print(f"Accuracy: {rb_metrics['accuracy']:.4f}")
            print(f"Precision: {rb_metrics['precision']:.4f}")
            print(f"Recall: {rb_metrics['recall']:.4f}")
            print(f"F1-Score: {rb_metrics['f1_score']:.4f}")
            print(f"F{beta}-Score: {rb_metrics[f'f{beta}_score']:.4f}")

            print("\n================== HASIL EVALUASI PER TAG (RULE-BASED) ==================")
            sorted_tags = sorted(tag_metrics.items(), key=lambda x: x[1]['support'], reverse=True)
            print(f"{'TAG':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Benar':>10}")
            print("-" * 70)
            for tag, metrics in sorted_tags:
                benar = rb_correct_count[tag]
                print(f"{tag:<10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['support']:>10d} {benar:>10d}")
        else:
            print("\nTidak ada kata yang di-tag oleh rule-based atau tidak ada data validasi.")

        # PERBAIKAN: Simpan hasil dengan method yang lebih detail
        os.makedirs("hasil/hybrid/data_2_baru", exist_ok=True)

        if y_true:
            # Simpan hasil tag yang salah dari keseluruhan
            wrong_file = "hasil/hybrid/data_2_baru/5k_baru/Hybrid_Wrong.txt"
            with open(wrong_file, 'w', encoding='utf-8') as f:
                f.write("No.\tKata\tPrediksi\tLabel_Benar\n")
                f.write("-" * 40 + "\n")
                for i, (word, pred, gold) in enumerate(zip(words, final_tags, y_true)):
                    if pred != gold:
                        f.write(f"{i+1}\t{word}\t{pred}\t{gold}\n")

            # Simpan hasil tag yang salah dari rule-based
            wrong_rb_file = "hasil/hybrid/data_2_baru/5k_baru/Hybrid_RuleBased_Wrong.txt"
            with open(wrong_rb_file, 'w', encoding='utf-8') as f:
                f.write("No.\tKata\tPrediksi\tLabel_Benar\n")
                f.write("-" * 40 + "\n")
                for idx, i in enumerate(rb_indices, 1):
                    word = words[i]
                    pred = final_tags[i]
                    gold = y_true[i]
                    if pred != gold:
                        f.write(f"{idx}\t{word}\t{pred}\t{gold}\n")

        # Simpan hasil tag berdasarkan jenis source
        output_dir = "hasil/hybrid/data_2_baru/5k_baru"
        source_files = {
            "hmm": open(os.path.join(output_dir, "HMM.txt"), "w", encoding="utf-8"),
            "smoothing": open(os.path.join(output_dir, "Smoothing.txt"), "w", encoding="utf-8"),
            "rule_based": open(os.path.join(output_dir, "Rule-Based.txt"), "w", encoding="utf-8"),
            "rule_based_selected": open(os.path.join(output_dir, "Rule-Based-Selected.txt"), "w", encoding="utf-8"),
            "lexicon": open(os.path.join(output_dir, "Lexicon.txt"), "w", encoding="utf-8"),
        }

        try:
            for i, (word, tag, source) in enumerate(zip(words, final_tags, tag_sources), 1):
                if source in source_files:
                    source_files[source].write(f"{i}. {word} -> {tag} (source: {source})\n")
                elif source.startswith("rule_based"):
                    source_files["rule_based"].write(f"{i}. {word} -> {tag} (source: {source})\n")

            # Simpan hasil tag keseluruhan dengan format kata|TAG
            output_tagged_file = "hasil/hybrid/data_2_baru/5k_baru/Hybrid_Tagged_Output.txt"
            with open(output_tagged_file, "w", encoding="utf-8") as f:
                for word, tag in zip(words, final_tags):
                    f.write(f"{word}|{tag} ")
                f.write("\n")

            print(f"\n\nHasil tag keseluruhan telah disimpan di {output_tagged_file}")
            print("Hasil tag HMM hasil/hybrid/data_2_baru/5k_baru/HMM.txt")
            print("Hasil tag smoothing hasil/hybrid/data_2_baru/5k_baru/Smoothing.txt")
            print("Hasil tag rule-based hasil/hybrid/data_2_baru/5k_baru/Rule-based.txt")
            print("Hasil tag rule-based terpilih hasil/hybrid/data_2_baru/5k_baru/Rule-based-selected.txt")
            print("Hasil tag lexicon hasil/hybrid/data_2_baru/5k_baru/Lexicon.txt")
            print(f"Hasil tag yang salah dari keseluruhan di {wrong_file}")
            print(f"Hasil tag yang salah dari rule-based di {wrong_rb_file}")

        finally:
            # Tutup semua file
            for f in source_files.values():
                f.close()


# if __name__ == "__main__":
#     # File training dan validasi (ubah sesuai nama dan lokasi file kamu)
#     training_file = "dataset/Korpus_Train.txt"
#     validation_file = "data/korpus_validasi5.txt"
#     input_file = "data/data_input5.txt"

#     # Membuat folder log
#     os.makedirs("logs", exist_ok=True)
#     log_file = "logs/hybrid_pos_log_uji.txt"
#     output_capture = TeeStringIO(log_file)
#     sys.stdout = output_capture

#     # Inisialisasi sistem POS Tagger hybrid
#     hybrid_tagger = HybridPOSTagger()

#     # Training HMM (hanya perlu dilakukan sekali)
#     hybrid_tagger.train_hmm(training_file)

#     # Baca input_text dari file input_file
#     with open(input_file, "r", encoding="utf-8") as f:
#         input_text = f.read().strip()

#     # Tag kalimat input
#     results = hybrid_tagger.tag_sentence(input_text)

#     # Evaluasi dan simpan hasil
#     output_dir = "hasil/uji/hybrid/uji"
#     hybrid_tagger.evaluate_and_save_results(results, validation_file, output_dir)

#     print(f"== Log File {log_file} ==")

#     # Kembalikan stdout ke normal dan tutup log
#     sys.stdout = output_capture.original_stdout
#     output_capture.close()

# if __name__ == "__main__":
#     # File training dan validasi (ubah sesuai nama dan lokasi file kamu)
#     training_file = "dataset/Korpus_Train.txt"
#     validation_file = "data/korpus_validasi_ID_5K.txt"
#     input_file = "data/data_input22.txt"

#     # Membuat folder log
#     os.makedirs("logs", exist_ok=True)
#     log_file = "logs/hybrid_pos_log_11.txt"
#     output_capture = TeeStringIO(log_file)
#     sys.stdout = output_capture

#     # Inisialisasi sistem POS Tagger hybrid
#     hybrid_tagger = HybridPOSTagger()

#     # Training HMM (hanya perlu dilakukan sekali)
#     hybrid_tagger.train_hmm(training_file)

#     # Baca input_text dari file input_file
#     with open(input_file, "r", encoding="utf-8") as f:
#         input_text = f.read().strip()

#     # Tag kalimat input
#     results = hybrid_tagger.tag_sentence(input_text)

#     # Evaluasi dan simpan hasil
#     output_dir = "hasil/hybrid/data_2_baru_baru"
#     hybrid_tagger.evaluate_and_save_results(results, validation_file, output_dir)

#     import openpyxl

#     # Ambil data OOV dan hasil tagging
#     words = results['words']
#     final_tags = results['final_tags']
#     hmm_tags = results['hmm_tags']
#     oov_indices = results['oov_stats']['oov_words']  # List of tuples (i, word, tag, status)
#     tag_sources = results['tag_sources']

#     # Ambil groundtruth
#     def get_groundtruth_labels(file_path):
#         y_true = []
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 line = line.strip()
#                 for word_label in line.split():
#                     if "|" in word_label:
#                         _, label = word_label.rsplit('|', 1)
#                         y_true.append(label)
#         return y_true

#     y_true = get_groundtruth_labels(validation_file)

#     # List OOV: index, word, hybrid_tag, hmm_tag, groundtruth_tag
#     oov_list = []
#     for oov_info in oov_indices:
#         idx = oov_info[0]
#         word = words[idx]
#         hybrid_tag = final_tags[idx]
#         hmm_tag = hmm_tags[idx]
#         gt_tag = y_true[idx] if idx < len(y_true) else ""
#         oov_list.append((word, hybrid_tag, hmm_tag, gt_tag))

#     # Simpan ke TXT
#     with open(os.path.join(output_dir, "OOV_Comparison_5k.txt"), "w", encoding="utf-8") as f:
#         f.write("Word\tHybrid_Tag\tHMM_Tag\tGroundtruth_Tag\n")
#         for word, hybrid_tag, hmm_tag, gt_tag in oov_list:
#             f.write(f"{word}\t{hybrid_tag}\t{hmm_tag}\t{gt_tag}\n")

#     # Simpan ke Excel
#     wb = openpyxl.Workbook()
#     ws = wb.active
#     ws.title = "OOV Comparison"
#     ws.append(["Word", "Hybrid_Tag", "HMM_Tag", "Groundtruth_Tag"])
#     for word, hybrid_tag, hmm_tag, gt_tag in oov_list:
#         ws.append([word, hybrid_tag, hmm_tag, gt_tag])
#     wb.save(os.path.join(output_dir, "OOV_Comparison_5k.xlsx"))

#     print(f"Daftar OOV dan perbandingan tag telah disimpan di {os.path.join(output_dir, 'OOV_Comparison_5k.txt')} dan .xlsx")

#     print(f"== Log File {log_file} ==")

#     # Kembalikan stdout ke normal dan tutup log
#     sys.stdout = output_capture.original_stdout
#     output_capture.close()
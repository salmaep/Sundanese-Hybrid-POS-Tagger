from ast import Not
import json
import re
import os
from matplotlib.cbook import flatten
from collections import defaultdict

class RuleBased:
    def __init__(self, affix_rule_file="dataset/Comprehensive.json"):
        self.lexicon = self.load_lexicon("dataset/Leksikon.txt")
        self.affix_rules = self.load_affix_rules(affix_rule_file)
        self.sintaksis_rules = self.load_syntax_rules("dataset/aturan_sintaksis.json")
        self.affix_rule_usage = defaultdict(list)
        self.transition_probs = self.load_transition_probabilities("probs_hmm_sunda.txt")

    def load_lexicon(self, filepath):
        lexicon = {}
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    word, label = parts
                    lexicon[word.lower()] = label
        return lexicon

    def load_affix_rules(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)

    def load_transition_probabilities(self, filepath):
        """Load transition probabilities from HMM model file"""
        transition_probs = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4 and parts[0] == 'T':
                        # Format: T prev_tag next_tag probability
                        prev_tag, next_tag, prob = parts[1], parts[2], float(parts[3])
                        transition_key = f"{prev_tag} {next_tag}"
                        transition_probs[transition_key] = prob
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Transition probability selection disabled.")
        return transition_probs

    def get_best_rule_by_transition(self, matching_rules, prev_tag="<s>"):
        """
        Select the best rule from multiple matching rules based on transition probability
        """
        if not matching_rules:
            return None, "UNK"
        
        if len(matching_rules) == 1:
            return matching_rules[0]
        
        # If multiple rules match, select based on transition probability
        best_rule = None
        best_prob = -1
        best_result = "UNK"
        
        print(f"  Multiple rules found ({len(matching_rules)}), selecting based on transition probability:")
        
        for rule_info, result_label in matching_rules:
            transition_key = f"{prev_tag} {result_label}"
            prob = self.transition_probs.get(transition_key, 0.0)
            
            print(f"    Rule: {rule_info} → {result_label}, Transition P({prev_tag}→{result_label}) = {prob:.6f}")
            
            if prob > best_prob:
                best_prob = prob
                best_rule = rule_info
                best_result = result_label
        
        print(f"  → Selected: {best_rule} → {best_result} (highest transition prob: {best_prob:.6f})")
        return best_rule, best_result

    def apply_affix_rules(self, prefix, suffix, infix, root_label, prev_tag="<s>", original_word=None):
        """
        Apply affix rules considering prefix, suffix, and infix
        Returns tuple of (rule_info, result_label)
        """
        print(f"Checking affix rules for prefix:'{prefix}', suffix:'{suffix}', infix:'{infix}', root_label:'{root_label}'")
        
        # Handle case where affixes might be None
        prefix = prefix or ""
        suffix = suffix or ""
        infix = infix or ""
        
        matching_rules = []
        
        for i, rule in enumerate(self.affix_rules):
            prefix_match = rule.get("prefix", "") == prefix
            suffix_match = rule.get("suffix", "") == suffix
            infix_match = rule.get("infix", "") == infix
            label_match = rule.get("root_label") == root_label
            
            # Debug info
            if prefix_match and suffix_match and infix_match and label_match:
                rule_info = f"Rule #{i} {prefix}-{infix}-{suffix}|{root_label}"
                result_label = rule["result_label"]
                matching_rules.append((rule_info, result_label))
                
                print(f"  Rule #{i}: prefix:{rule.get('prefix', '')}, suffix:{rule.get('suffix', '')}, " +
                      f"infix:{rule.get('infix', '')}, root_label:{rule.get('root_label')}, " +
                      f"result:{result_label}")
        
        if not matching_rules:
            print("  No matching rule found")
            return "No rule matched", "UNK"
        
        # Select best rule based on transition probability
        best_rule_info, best_result_label = self.get_best_rule_by_transition(matching_rules, prev_tag)
        
        if best_result_label != "UNK":
            # Record rule usage for the selected rule
            for i, rule in enumerate(self.affix_rules):
                rule_info_check = f"Rule #{i} {prefix}-{infix}-{suffix}|{root_label}"
                if rule_info_check == best_rule_info:
                    # self.affix_rule_usage[i].append(f"{prefix}-{infix}-{suffix}|{root_label}")
                    self.affix_rule_usage[i].append((original_word.lower(), f"{prefix}-{infix}-{suffix}|{root_label}"))
                    break
        
        return best_rule_info, best_result_label
    
    def load_syntax_rules(self, filepath):
        if not os.path.exists(filepath):
            print(f"File {filepath} tidak ditemukan.")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # def isRootWord(self, word):
    #     if re.search(" " + re.escape(word) + " ", self.rootWords):
    #         return 1
    #     else:
    #         return 0

    def isRootWord(self, word):
        return 1 if word.lower() in self.lexicon else 0

    def tokenize(self, sentence):
        return sentence.lower().split()

    def writeLog(self, word, base, affix):
        with open('log/log.txt', 'a+', encoding='utf-8') as f:
            f.write(f'{word} -> {base}, Affix: {affix}\n')

    def prefixRemoval(self, word, history):
        temp = []
        prefixTemp = []
        while True:
            if word.startswith(("dipika")):
                prefixTemp.append(word[:6])
                history.append((word, word[6:], word[:6], "prefix"))
                word = word[6:]
            elif word.startswith(("nyang")):
                prefixTemp.append(word[:5])
                history.append((word, word[5:], word[:5], "prefix"))
                word = word[5:]
            elif word.startswith(("pang", "mang", "sang", "ting", "dipi", "pari")):
                prefixTemp.append(word[:4])
                history.append((word, word[4:], word[:4], "prefix"))
                word = word[4:]
            elif word.startswith(("per", "nga", "nge", "pra", "pri")):
                prefixTemp.append(word[:3])
                history.append((word, word[3:], word[:3], "prefix"))
                word = word[3:]
            elif word.startswith(("mi", "ma", "ba", "sa", "si", "di", "ka", "pa", "pi", "ti", "an")):
                prefixTemp.append(word[:2])
                history.append((word, word[2:], word[:2], "prefix"))
                word = word[2:]
            elif word.startswith("a"):
                prefixTemp.append("a")
                history.append((word, word[1:], "a", "prefix"))
                word = word[1:]
            else:
                break
            temp.append(word)
        return temp, prefixTemp

    def suffixRemoval(self, word, history):
        temp = []
        suffixTemp = []
        while True:
            if word.endswith(("keunana", "eunana", "anana")):
                suffixTemp.append(word[-6:])
                history.append((word, word[:-6], word[-6:], "suffix"))
                word = word[:-6]
            elif word.endswith(("keun", "ning", "ting")):
                suffixTemp.append(word[-4:])
                history.append((word, word[:-4], word[-4:], "suffix"))
                word = word[:-4]
            elif word.endswith(("eun", "ing")):
                suffixTemp.append(word[-3:])
                history.append((word, word[:-3], word[-3:], "suffix"))
                word = word[:-3]
            elif word.endswith(("na", "an")):
                suffixTemp.append(word[-2:])
                history.append((word, word[:-2], word[-2:], "suffix"))
                word = word[:-2]
            else:
                break
            temp.append(word)
        return temp, suffixTemp

    def allomorphNormalize(self, word, history):
        results = []
        if word.startswith("ng"):
            str1 = word.replace("ng", "k", 1)
            history.append((word, str1, "ng->k", "allomorph"))
            results.append(str1)
            
            str2 = word.replace("ng", "g", 1)
            history.append((word, str2, "ng->g", "allomorph"))
            results.append(str2)
            
            str3 = word.replace("ng", "p", 1)
            history.append((word, str3, "ng->p", "allomorph"))
            results.append(str3)
            
            return results, "ng"
        
        if word.startswith("ny"):
            str1 = word.replace("ny", "c", 1)
            history.append((word, str1, "ny->c", "allomorph"))
            results.append(str1)
            
            str2 = word.replace("ny", "s", 1)
            history.append((word, str2, "ny->s", "allomorph"))
            results.append(str2)
            
            return results, "ny"
        
        if word.startswith("m"):
            str1 = word.replace("m", "b", 1)
            history.append((word, str1, "m->b", "allomorph"))
            results.append(str1)
            
            str2 = word.replace("m", "p", 1)
            history.append((word, str2, "m->p", "allomorph"))
            results.append(str2)
            
            return results, "m"
        
        if word.startswith("n"):
            str1 = word.replace("n", "t", 1)
            history.append((word, str1, "n->t", "allomorph"))
            results.append(str1)
            
            return results, "n"
        
        return [], ""

    def InfixRemoval(self, word, history):
        temp = []
        infixTemp = []
        while True:
            if "ar" in word and word.index("ar") < 3:
                temp.append(word.replace("ar", "", 1))
                infixTemp.append("ar")
                history.append((word, word.replace("ar", "", 1), "ar", "infix"))
                word = word.replace("ar", "", 1)
            elif "al" in word and word.index("al") < 3:
                temp.append(word.replace("al", "", 1))
                infixTemp.append("al")
                history.append((word, word.replace("al", "", 1), "al", "infix"))
                word = word.replace("al", "", 1)
            elif "in" in word and word.index("in") < 3:
                temp.append(word.replace("in", "", 1))
                infixTemp.append("in")
                history.append((word, word.replace("in", "", 1), "in", "infix"))
                word = word.replace("in", "", 1)
            elif "um" in word and word.index("um") < 3:
                temp.append(word.replace("um", "", 1))
                infixTemp.append("um")
                history.append((word, word.replace("um", "", 1), "um", "infix"))
                word = word.replace("um", "", 1)
            else:
                break
        return temp, infixTemp

    def reduplicationNormalize(self, word):
        """
        Enhanced reduplication normalization with tracking capability
        Returns the normalized word and information about reduplication type
        """
        if (word[0:1] == word[1:2]):
            return word[1:], "single_char"
        if (word[0:2] == word[2:4]):
            return word[2:], "double_char"
        if (word[0:3] == word[3:6]):
            return word[3:], "triple_char"
        if (word.find("ng", 1, (len(word) // 2)) != -1):
            pos = word.find(("ng"))
            if (word[:pos] == word[pos + 2:pos + 2 + pos]):
                return word[pos + 2:], "ng_reduplication"
        if (len(word) > 5) and (word[2:4] == word[4:6]):
            return word[0:2] + word[4:], "middle_reduplication"
        if (len(word) > 7) and (word[2:5] == word[5:8]):
            return word[0:2] + word[5:], "complex_reduplication"
        
        # Check for hyphenated reduplication (like "imah-imah")
        if "-" in word:
            parts = word.split("-")
            if len(parts) == 2 and parts[0] == parts[1]:
                return parts[0], "hyphenated_exact"
        
        return None, None

    def check_exact_reduplication(self, original_word, stemmed_word):
        """
        Check if the original word is an exact reduplication of the stemmed word
        Returns True if it's exact reduplication, False otherwise
        """
        if "-" in original_word:
            parts = original_word.split("-")
            if len(parts) == 2 and parts[0] == parts[1] and parts[0] == stemmed_word:
                return True
        
        # Check other reduplication patterns
        normalized, reduplicate_type = self.reduplicationNormalize(original_word)
        if normalized == stemmed_word and reduplicate_type in ["single_char", "double_char", "triple_char", "hyphenated_exact"]:
            return True
            
        return False

    def prefixProcess(self, word_list, history):
        resultList = []
        prefixList = []

        for word in word_list:
            remaining_word, prefixTemp = self.prefixRemoval(word, history)
            if prefixTemp:
                resultList.append(remaining_word)
                prefixList.extend(prefixTemp)

        return resultList, prefixList

    def suffixProcess(self, word_list, history):
        resultList = []
        suffixList = []

        for word in word_list:
            remaining_word, suffixTemp = self.suffixRemoval(word, history)
            if suffixTemp:
                resultList.append(remaining_word)
                suffixList.extend(suffixTemp)

        return resultList, suffixList
    
    def infixProcess(self, word_list, history):
        resultList = []
        infixList = []

        for word in word_list:
            remaining_word, infixTemp = self.InfixRemoval(word, history)
            if infixTemp:
                resultList.append(remaining_word)
                infixList.extend(infixTemp)

        print(f"DEBUG: Infix Process - Result List: {resultList}, Infix List: {infixList}")
        return resultList, infixList

    def allomorphProcess(self, word_list, history):
        resultList = []
        allomorphMap = {}

        for word in word_list:
            allomorph_forms, nasal = self.allomorphNormalize(word, history)
            if allomorph_forms:
                resultList.extend(allomorph_forms)
                for form in allomorph_forms:
                    allomorphMap[form] = {"original_word": word, "nasal": nasal, "replaced_letter": form[0]}
                
                print(f"DEBUG: Allomorph {word} -> {allomorph_forms} with nasal {nasal}")
        
        return resultList, allomorphMap

    def find_removed_affix(self, root_word, infixList):
        word = ""
        affix = ""
        for i in range(len(infixList) - 1):
            if isinstance(infixList[i], list) and root_word in infixList[i]:
                word = infixList[i - 1]
                affix = ''.join(infixList[i + 1])
        return word, affix

    def trace_back_affixes(self, root, history):
        transformations = []
        current_word = root

        while True:
            found = False
            for entry in history:
                before, after, removed, affix_type = entry
                if after == current_word:
                    transformations.append((removed, affix_type))
                    current_word = before
                    found = True
                    break
            
            if not found:
                break

        transformations.reverse()
        return transformations

    def apply_nasal_rules(self, stemmed_word, allomorphMap, lexicon_label):
        """
        Apply nasal transformation rules when needed.
        """
        result_label = lexicon_label

        if stemmed_word in allomorphMap:
            nasal_info = allomorphMap[stemmed_word]
            nasal = nasal_info.get("nasal", "")
            replaced_letter = nasal_info.get("replaced_letter", "")
            label_upper = str(lexicon_label).upper()

            if label_upper == "VB":
                if (nasal == "m" and replaced_letter == "b"):
                    print(f"Applying nasal rule: {nasal}->{replaced_letter} with VB tag remains VB")
                    result_label = "VB"

            elif label_upper == "NN":
                if (nasal == "ny" and replaced_letter == "c") or \
                   (nasal == "ng" and replaced_letter == "k") or \
                   (nasal == "m" and replaced_letter in ("b", "p")):
                    print(f"Applying nasal rule: {nasal}->{replaced_letter} with NN tag becomes VB")
                    result_label = "VB"

        return result_label

    def stemmingProcess(self, word):
        history = []
        allomorphMap = {}

        if self.isRootWord(word):
            self.writeLog(word, word, "rootWord")
            return word, history, allomorphMap
        else:
            stemList = []
            stemListTemp = []
            prefixList = []
            suffixList = []
            infixList = []

            stemWord = word.split('-')[-1]
            stemListTemp.append(stemWord)
            stemList.append(stemWord)

            # 1. Prefix
            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            for _ in range(5):
                stem, prefixTemp = self.prefixProcess(stemListTemp, history)
                prefixList.extend(prefixTemp)
                if not stem:
                    break
                stemListTemp = list(dict.fromkeys(flatten(stem)))
                stemList.extend(stemListTemp)

            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            for x in stemListTemp:
                stem, reduplicate_type = self.reduplicationNormalize(x)
                if stem:
                    stemList.append(stem)
                    # Track reduplication in history for later use
                    if reduplicate_type:
                        history.append((x, stem, reduplicate_type, "reduplication"))
            
            # 2. Suffix
            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            for _ in range(4):
                stem, suffixTemp = self.suffixProcess(stemListTemp, history)
                suffixList.extend(suffixTemp)
                if not stem:
                    break
                stemListTemp = list(dict.fromkeys(flatten(stem)))
                stemList.extend(stemListTemp)

            # 3. Infix
            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            for x in stemListTemp:
                stem, infixTemp = self.InfixRemoval(x, history)
                if stem:
                    infixList.extend(infixTemp)
                    stemList.extend(stem)
            
            # 4. Allomorph
            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            allomorphResults, allomorphMapTemp = self.allomorphProcess(stemListTemp, history)
            if allomorphResults:
                stemList.extend(allomorphResults)
                allomorphMap.update(allomorphMapTemp)

                for form in allomorphResults:
                    if self.isRootWord(form):
                        print("\n")
                        print(f"Root word found after allomorph: {form}")
                        transformations = self.trace_back_affixes(form, history)
                        print("Transformations:")
                        for removed, affix_type in transformations:
                            print(f"{affix_type} : {removed}")
                        self.writeLog(word, form, f"Transformations: {transformations}")
                        return form, history, allomorphMap

            # Cari kata dasar dari semua hasil
            stemListTemp = list(dict.fromkeys(flatten(stemList)))
            for stem in stemListTemp:
                if self.isRootWord(stem):
                    print("\n")                   
                    transformations = self.trace_back_affixes(stem, history)
                    print("Transformations:")
                    for removed, affix_type in transformations:
                        print(f"{affix_type} : {removed}")
                    self.writeLog(word, stem, f"Transformations: {transformations}")
                    return stem, history, allomorphMap

            self.writeLog(word, word, "UNK")
            return word, history, allomorphMap

    
    def is_probable_tag(self, word, expected_label):
        return expected_label in self.lexicon.get(word, []) or word not in self.lexicon
    
    def is_numeric_token(self, token):
        cleaned = token.replace(",", "").replace(".", "")
        return cleaned.isdigit() and token[0].isdigit()
    
    def check_lexicon_and_rules(self, word, prev_word="", next_word="", prev_tag="<s>"):
        """
        Check word against lexicon and rules with proper method tracking.
        Returns tuple (root_word, tag) for compatibility with existing code.
        """
        # Reset tracking attributes
        self.last_applied_rule = None
        self.last_method_used = None
        
        print(f"\n=== Analyzing word: '{word}' ===")

        # 1. Check if number
        if self.is_numeric_token(word):
            print(f"✓ {word} identified as number → tagged CD")
            self.last_method_used = "numeric"
            return word, "CD"

        # 2. Direct lexicon lookup
        if word in self.lexicon:
            tag = self.lexicon[word]
            print(f"✓ Found '{word}' in lexicon → {tag}")
            self.last_method_used = "lexicon"
            return word, tag

        # 3. Stemming and affix rules
        print(f"Attempting stemming for '{word}'...")
        history = []
        stemmed_word, stem_history, allomorphMap = self.stemmingProcess(word)
        print(f"Stemming result: {word} → {stemmed_word}")

        if stemmed_word in self.lexicon:
            root_label = self.lexicon[stemmed_word]
            print(f"✓ Stemmed word '{stemmed_word}' found in lexicon → {root_label}")
        
            # NEW: Check for exact reduplication
            if self.check_exact_reduplication(word, stemmed_word):
                print(f"✓ Exact reduplication detected: '{word}' is reduplication of '{stemmed_word}'")
                print(f"✓ Applying lexicon tag from stemmed word: {root_label}")
                self.last_method_used = "exact_reduplication"
                self.last_applied_rule = f"Exact reduplication: {word} → {stemmed_word}"
                return stemmed_word, root_label
        
            # Extract affixes from history
            transformations = self.trace_back_affixes(stemmed_word, stem_history)
            prefixes = []
            suffixes = []
            infixes = []
            for removed, affix_type in transformations:
                if affix_type == "prefix":
                    prefixes.append(removed)
                elif affix_type == "suffix":
                    suffixes.append(removed)
                elif affix_type == "infix":
                    infixes.append(removed)
            
            prefix = ''.join(prefixes)
            suffix = ''.join(reversed(suffixes))
            infix = ''.join(infixes)
            print(f"Extracted affixes: prefix='{prefix}', suffix='{suffix}', infix='{infix}'")

            # Apply affix rules
            rule_info, result_label = self.apply_affix_rules(prefix, suffix, infix, root_label, prev_tag=prev_tag, original_word=word)
            if result_label != "UNK":
                print(f"✓ Affix rule applied: {rule_info} → {result_label}")
                self.last_method_used = "affix_rules"
                self.last_applied_rule = rule_info
                return stemmed_word, result_label

        # 4. Nasal rules
        if stemmed_word in allomorphMap:
            root_label = self.lexicon.get(stemmed_word, "UNK")
            new_label = self.apply_nasal_rules(stemmed_word, allomorphMap, root_label)

            if new_label != "UNK" and prefix == '' and suffix == '' and infix == '':
                self.last_method_used = "nasal_rules"
                print(f"✓ Nasal rule applied: {root_label} → {new_label}")
                return stemmed_word, new_label
            else:
                root_label = new_label  # Untuk proses selanjutnya

        # 5. Syntactic rules
        print("Checking syntactic rules...")
        next_word_label = self.lexicon.get(next_word)
        prev_word_label = self.lexicon.get(prev_word)

        for rule in self.sintaksis_rules:
            target = rule.get("target")
            trigger_word = rule.get("trigger_word")
            trigger_label = rule.get("trigger_label")
            expected_label = rule.get("expected_label")

            # TARGET: 'next' → Cek kata sebelumnya
            if target == "next":
                if (prev_word == trigger_word) or (prev_word_label == trigger_label and prev_word == trigger_word):
                    print(f"✓ Rule match for 'next': prev word '{prev_word}' matches trigger")
                    print(f"↪ Assigning expected label to '{word}' → {expected_label}")
                    self.last_method_used = "syntax_rules"
                    self.last_applied_rule = f"Trigger in prev_word → {expected_label}"
                    return word, expected_label

            # TARGET: 'prev' → Cek kata sesudahnya
            elif target == "prev":
                if (next_word == trigger_word) or (next_word == trigger_word and next_word_label == trigger_label):
                    print(f"✓ Rule match for 'prev': next word '{next_word}' matches trigger")
                    print(f"↪ Assigning expected label to '{word}' → {expected_label}")
                    self.last_method_used = "syntax_rules"
                    self.last_applied_rule = f"Trigger in next_word → {expected_label}"
                    return word, expected_label

        print("✗ No rules matched, returning UNK")
        self.last_method_used = "unknown"
        return word, "UNK"

    def get_method_summary(self):
        """
        Get summary of the last method used for debugging purposes
        """
        return {
            "method": self.last_method_used,
            "rule": self.last_applied_rule
        }
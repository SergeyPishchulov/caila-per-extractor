import re


class AproxMatcher:
    @staticmethod
    def levenshtein_distance(s, t):
        m = len(s)
        n = len(t)
        d = [[0] * (n + 1) for i in range(m + 1)]

        for i in range(1, m + 1):
            d[i][0] = i

        for j in range(1, n + 1):
            d[0][j] = j

        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s[i - 1] == t[j - 1]:
                    cost = 0
                else:
                    cost = 1
                d[i][j] = min(d[i - 1][j] + 1,  # deletion
                              d[i][j - 1] + 1,  # insertion
                              d[i - 1][j - 1] + cost)  # substitution

        return d[m][n]

    def find(self, string: str, text: str):
        string = ' '.join(string.rstrip('.').split())
        text = ' '.join(text.rstrip('.').split())

        words_cnt_in_string = len(string.split())

        text_word_lens = [len(word) for word in text.split()]
        shift = 0
        res = []
        for word_len in text_word_lens:
            dist = self.levenshtein_distance(string, text[shift:shift + len(string)])
            if dist / len(string) < 0.25:
                words = [w.rstrip('.') if len(w) > 2 else w for w in text[shift:].split()]
                extracted_name = ' '.join(words[:words_cnt_in_string])
                starts_ends = [(m.start(), m.end()) for m in re.finditer(extracted_name, text)]
                res.extend(starts_ends)
            shift += word_len + 1
        return res

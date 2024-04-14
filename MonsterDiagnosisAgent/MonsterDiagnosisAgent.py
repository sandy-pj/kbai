from itertools import combinations
class MonsterDiagnosisAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def solve(self, diseases, patient):
        # Add your code here!
        #
        # The first parameter to this method is a list of diseases, represented as a
        # dictionary. The key in each dictionary entry is the name of a disease. The
        # value for the key item is another dictionary of symptoms of that disease, where
        # the keys are letters representing vitamin names ("A" through "Z") and the values
        # are "+" (for elevated), "-" (for reduced), or "0" (for normal).
        #
        # The second parameter to this method is a particular patient's symptoms, again
        # represented as a dictionary where the keys are letters and the values are
        # "+", "-", or "0".
        #
        # This method should return a list of names of diseases that together explain the
        # observed symptoms. If multiple lists of diseases can explain the symptoms, you
        # should return the smallest list. If multiple smallest lists are possible, you
        # may return any sufficiently explanatory list.
        #
        # The solve method will be called multiple times, each of which will have a new set
        # of diseases and a new patient to diagnose.
        #

        # generate all possible combinations of diseases
        MAX_DISEASES = 7
        all_combinations = []
        for i in range(1, MAX_DISEASES+1):
            all_combinations.extend(combinations(diseases.keys(), i))

        # translate the combination into symptoms
        for combination in all_combinations:
            symptoms = {}
            for disease in combination:
                disease_symptoms = diseases[disease]
                for vitamin, value in disease_symptoms.items():
                    if vitamin not in symptoms:
                        symptoms[vitamin] = 0
                    if value == "+":
                        symptoms[vitamin] += 1
                    elif value == "-":
                        symptoms[vitamin] -= 1

            # translate back to +-0
            for vitamin, value in symptoms.items():
                if value > 0:
                    symptoms[vitamin] = "+"
                elif value < 0:
                    symptoms[vitamin] = "-"
                else:
                    symptoms[vitamin] = "0"

            # check if the symptoms match the patient's symptoms
            matched = self.match_symptoms(symptoms, patient)
            if matched:
                return combination
            else:
                pass
        return []

    def match_symptoms(self, symptoms, patient):
        mismatch_cnt = 0
        for k, v in patient.items():
            if patient[k] != symptoms[k]:
                mismatch_cnt += 1
        print("mismatch_cnt: \n", mismatch_cnt)
        if mismatch_cnt == 0:
            return True
        else:
            return False

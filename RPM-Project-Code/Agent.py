# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
import os
from PIL import Image, ImageChops
import numpy as np
from enum import Enum
import cv2
from ProblemSet import ProblemSet

DEBUG = 1
def normal_pdf(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-1 * (x - mu)**2 / (2 * sigma**2))

def l2distance(np_image1, np_image2):
    return np.linalg.norm(np_image1 - np_image2)
class Shape(Enum):
    triangle = 1,
    rectangle = 2,
    circle = 3,
    hexagon = 4

class Transformation(Enum):
    fill = 1,
    delete = 2,
    add = 3,
    change = 4
    flip = 5,
    rotate = 6

class TransformationSuggester:
    def __init__(self):
        pass

    def _flip(self, np_image, axis = 0):
        return np.flip(np_image, axis)
    def _fill(self, np_image, color = 0):
        return np_image.fill(color)

    def _rotate(self, np_image, angle = 90):
        return np.rot90(np_image, k = int( angle /90))

    def suggest_transformation(self, np_image1, np_image2):
        # Compare the two images and suggest a transformation
        res = ()
        min_distance = float('inf')
        for trans in Transformation:
            if trans == Transformation.rotate:
                for angle in [90, 180, 270]:
                    _d = l2distance(self._rotate(np_image1, angle), np_image2)
                    if _d < min_distance:
                        min_distance = _d
                        res = (TransformationSuggester._rotate, {'angle': angle})
            elif trans == Transformation.flip:
                for axis in [0, 1]:
                    _d = l2distance(self._flip(np_image1, axis), np_image2)
                    if _d < min_distance:
                        min_distance = _d
                        res = (TransformationSuggester._flip, {'axis': axis})
        return res

    def apply_transformation(self, np_image, func_trans, *args):
        res_image = func_trans(self, np_image, **args[0])
        # Image.fromarray(res_image).save('/tmp/transformed.png')
        return res_image



class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    problem = None

    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    H = []

    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []


    ops = {'AB': {'action': None},
           'AC': {'action': None},
           'BC': {'action': None}, # 3x3 horizontal
           'DG': {'action': None}} # 3x3 vertical

    answers = []

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.

    def getIdealImage(self, problem):
        # Get the ideal image from the problem
        images = self.dataset_from_problem(problem)
        if problem.problemType == '2x2':
            t = TransformationSuggester()
            self.ops['AB']['action'] = t.suggest_transformation(images['A'], images['B'])
            self.ops['AC']['action'] = t.suggest_transformation(images['A'], images['C'])
            if DEBUG: print("AB: %s, AC: %s" % (self.ops['AB']['action'], self.ops['AC']['action']))
            idealImages = []
            idealImages.append(t.apply_transformation(images['C'], self.ops['AB']['action'][0], self.ops['AB']['action'][1]))
            idealImages.append(t.apply_transformation(images['B'], self.ops['AC']['action'][0], self.ops['AC']['action'][1]))
            # [Image.fromarray(idealImages[_]).save('/tmp/idealImage_%s.png' % _ ) for _ in range(len(idealImages))]
            return idealImages
        elif problem.problemType == '3x3':
            t = TransformationSuggester()
            self.ops['BC']['action'] = t.suggest_transformation(images['B'], images['C'])
            self.ops['DG']['action'] = t.suggest_transformation(images['D'], images['G'])
            if DEBUG: print("BC: %s, DG: %s" % (self.ops['BC']['action'], self.ops['DG']['action']))
            idealImages = []
            idealImages.append(t.apply_transformation(images['H'], self.ops['BC']['action'][0], self.ops['BC']['action'][1]))
            idealImages.append(t.apply_transformation(images['F'], self.ops['DG']['action'][0], self.ops['DG']['action'][1]))
            return idealImages

    def find_best_match(self, problem, idealImages):
        # Find the best match
        best_matches = []
        images = self.dataset_from_problem(problem)

        min_distance_dict = {}
        for i in range(1, 9):
            if problem.problemType == '2x2' and i in [7,8]:
                continue
            min_distance = float('inf')
            for idealImage in idealImages:
                _d = l2distance(images[str(i)], idealImage)
                if _d < min_distance:
                    min_distance = _d
                    min_distance_dict[i] = min_distance
        best_matches = sorted(min_distance_dict, key=min_distance_dict.get, reverse=False)
        print(min_distance_dict) if DEBUG else None
        print("%s: Best matches %s" % (problem.name, best_matches))
        return best_matches

    def Solve(self, problem):
        self.problem = problem

        # Image processing
        images = self.dataset_from_problem(problem)

        self.answers = [images['1'], images['2'], images['3'], images['4'], images['5'], images['6']]
        if problem.problemType == "3x3":
            self.answers.append(images['7'])
            self.answers.append(images['8'])

        ########## problem set specific ##########
        if "Problems C" in problem.problemSetName:
            return self.Solve_w_metrics(problem, images)
        if "Challenge Problems D" in problem.problemSetName:
            return self.Solve_D(problem, images)
        if "Problems D" in problem.problemSetName:
            return self.Solve_w_metrics(problem, images)
        if "Problems E" in problem.problemSetName:
            return self.Solve_w_metrics(problem, images)
        if "Problems B" in problem.problemSetName:
            return self.Solve_w_metrics(problem, images)

    def Solve_v2(self, problem, images):
        if problem.problemType == '2x2':
            # A B
            # C D ====> D

            # Horizontal: AB -> CD
            # Vertical: AC -> BD
            dprAB = self._DPR(images['A'], images['B'])
            iprAB = self._IPR(images['A'], images['B'])
            dprAC = self._DPR(images['A'], images['C'])
            iprAC = self._IPR(images['A'], images['C'])
            dprCDs = [self._DPR(images['C'], img) for img in self.answers]
            iprCDs = [self._IPR(images['C'], img) for img in self.answers]

            next_round_roster = []
            for index in range(6):
                if np.abs(iprAB * iprCDs[index]) < 2 or np.abs(iprAC * iprCDs[index]) < 2:
                    next_round_roster.append(index)
            if len(next_round_roster) > 0:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in range(6)] ) + 1
            return choose
        if problem.problemType == "3x3":
            self.answers.append(images['7'])
            self.answers.append(images['8'])

        # Calculate the metrics
        # A B C
        # D E F
        # G H I

        # Horizontal: HI <- GH
        # Vertical: FI <- CF
        # Diagnal: EI <- AE

        ## the candidate image with the smallest abs dpr has the highest dpr score
        dprGH = self._DPR(images['G'], images['H'])
        iprGH = self._IPR(images['G'], images['H'])
        dprCF = self._DPR(images['C'], images['F'])
        iprCF = self._IPR(images['C'], images['F'])
        dprAE = self._DPR(images['A'], images['E'])
        iprAE = self._IPR(images['A'], images['E'])
        dprHIs = [self._DPR(images['H'], img) for img in self.answers]
        iprHIs = [self._IPR(images['H'], img) for img in self.answers]
        dprFIs = [self._DPR(images['F'], img) for img in self.answers]
        iprFIs = [self._IPR(images['F'], img) for img in self.answers]
        dprEIs = [self._DPR(images['E'], img) for img in self.answers]
        iprEIs = [self._IPR(images['E'], img) for img in self.answers]


        scores = [ \
                    normal_pdf(dprHIs[i], dprGH, 1)  + \
                    normal_pdf(iprHIs[i], iprGH, 1) + \
                    normal_pdf(dprFIs[i], dprCF, 1)  + \
                    normal_pdf(iprFIs[i], iprCF, 1) + \
                    normal_pdf(dprEIs[i], dprAE, 1)  + \
                    normal_pdf(iprEIs[i], iprAE, 1) \
            for i in range(8)
        ]


        choose = np.argmax(scores)+1

        return choose

    def Solve_w_metrics(self, problem, images):
        if problem.problemType == '2x2':
            # A B
            # C D ====> D

            # Horizontal: AB -> CD
            # Vertical: AC -> BD
            dprAB = self._DPR(images['A'], images['B'])
            iprAB = self._IPR(images['A'], images['B'])
            dprAC = self._DPR(images['A'], images['C'])
            iprAC = self._IPR(images['A'], images['C'])
            dprCDs = [self._DPR(images['C'], img) for img in self.answers]
            iprCDs = [self._IPR(images['C'], img) for img in self.answers]

            next_round_roster = []
            for index in range(6):
                if np.abs(iprAB * iprCDs[index]) < 2 or np.abs(iprAC * iprCDs[index]) < 2:
                    next_round_roster.append(index)
            if len(next_round_roster) > 0:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in range(6)] ) + 1
            return choose
        if problem.problemType == "3x3":
            self.answers.append(images['7'])
            self.answers.append(images['8'])

        # Calculate the metrics
        # A B C
        # D E F
        # G H I

        # Horizontal: HI <- BC
        # Vertical: FI <- CF

        # 1. DPR - Dark Pixel Ratio
        # 2. IPR - Intersection Pixel Ratio
        # 3. Filter with DPR threshold
        # 4. define scoring
        ## the candidate image with the smallest abs dpr has the highest dpr score
        dprBC = self._DPR(images['B'], images['C'])
        dprHIs = [self._DPR(images['H'], img) for img in self.answers]
        iprBC = self._IPR(images['B'], images['C'])
        iprHIs = [self._IPR(images['H'], img) for img in self.answers]

        scores = [( \
                   normal_pdf(d, dprBC, 1)  + \
                   normal_pdf(i, iprBC, 1)   \
                   )/2
                    if (i*iprBC>0 and d*dprBC>0) else 0 \
                  for d, i in zip(dprHIs, iprHIs)  ]

        choose = np.argmax(scores)+1

        print("%s: Best match %s" % (problem.name, choose))
        print("DPR %s: %s" % (dprBC, dprHIs)) if DEBUG else None
        print("IPR %s: %s" % (iprBC, iprHIs)) if DEBUG else None
        print("Sores: %s" % scores) if DEBUG else None
        print("\n")
        return choose

    def Solve_GH(self, problem, images):
        if problem.problemType == '2x2':
            # A B
            # C D ====> D

            # Horizontal: AB -> CD
            # Vertical: AC -> BD
            dprAB = self._DPR(images['A'], images['B'])
            iprAB = self._IPR(images['A'], images['B'])
            dprAC = self._DPR(images['A'], images['C'])
            iprAC = self._IPR(images['A'], images['C'])
            dprCDs = [self._DPR(images['C'], img) for img in self.answers]
            iprCDs = [self._IPR(images['C'], img) for img in self.answers]

            next_round_roster = []
            for index in range(6):
                if np.abs(iprAB * iprCDs[index]) < 2 or np.abs(iprAC * iprCDs[index]) < 2:
                    next_round_roster.append(index)
            if len(next_round_roster) > 0:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in range(6)] ) + 1
            return choose
        if problem.problemType == "3x3":
            self.answers.append(images['7'])
            self.answers.append(images['8'])

        # Calculate the metrics
        # A B C
        # D E F
        # G H I

        # Horizontal: HI <- BC
        # Vertical: FI <- CF
        # Diagnal: EI <- AE

        # 1. DPR - Dark Pixel Ratio
        # 2. IPR - Intersection Pixel Ratio
        # 3. Filter with DPR threshold
        # 4. define scoring
        ## the candidate image with the smallest abs dpr has the highest dpr score
        dprGH = self._DPR(images['G'], images['H'])
        dprHIs = [self._DPR(images['H'], img) for img in self.answers]
        iprGH = self._IPR(images['G'], images['H'])
        iprHIs = [self._IPR(images['H'], img) for img in self.answers]

        scores = [( \
                   normal_pdf(d, dprGH, 1)  + \
                   normal_pdf(i, iprGH, 1)   \
                   )/2
                    if (i*iprGH>0 and d*dprGH>0) else 0 \
                  for d, i in zip(dprHIs, iprHIs)  ]

        choose = np.argmax(scores)+1

        return choose
    def Solve_D(self, problem, images):
        if problem.problemType == '2x2':
            # A B
            # C D ====> D

            # Horizontal: AB -> CD
            # Vertical: AC -> BD
            dprAB = self._DPR(images['A'], images['B'])
            iprAB = self._IPR(images['A'], images['B'])
            dprAC = self._DPR(images['A'], images['C'])
            iprAC = self._IPR(images['A'], images['C'])
            dprCDs = [self._DPR(images['C'], img) for img in self.answers]
            iprCDs = [self._IPR(images['C'], img) for img in self.answers]

            next_round_roster = []
            for index in range(6):
                if np.abs(iprAB * iprCDs[index]) < 2 or np.abs(iprAC * iprCDs[index]) < 2:
                    next_round_roster.append(index)
            if len(next_round_roster) > 0:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in range(6)] ) + 1
            return choose
        if problem.problemType == "3x3":
            self.answers.append(images['7'])
            self.answers.append(images['8'])

        # Calculate the metrics
        # A B C
        # D E F
        # G H I

        # Horizontal: HI <- BC
        # Vertical: FI <- CF
        # Diagnal: EI <- AE

        # 1. DPR - Dark Pixel Ratio
        # 2. IPR - Intersection Pixel Ratio
        # 3. Filter with DPR threshold
        # 4. define scoring
        ## the candidate image with the smallest abs dpr has the highest dpr score
        dprAE = self._DPR(images['A'], images['E'])
        dprEIs = [self._DPR(images['E'], img) for img in self.answers]
        iprAE = self._IPR(images['A'], images['E'])
        iprEIs = [self._IPR(images['E'], img) for img in self.answers]

        scores = [( \
                   normal_pdf(d, dprAE, 1)  + \
                   normal_pdf(i, iprAE, 1)   \
                   )/2
                    if (i*iprAE>0 and d*dprAE>0) else 0 \
                  for d, i in zip(dprEIs, iprEIs)  ]

        choose = np.argmax(scores)+1

        return choose
    def Solve_C(self, problem, images):
        if problem.problemType == '2x2':
            # A B
            # C D ====> D

            # Horizontal: AB -> CD
            # Vertical: AC -> BD
            dprAB = self._DPR(images['A'], images['B'])
            iprAB = self._IPR(images['A'], images['B'])
            dprAC = self._DPR(images['A'], images['C'])
            iprAC = self._IPR(images['A'], images['C'])
            dprCDs = [self._DPR(images['C'], img) for img in self.answers]
            iprCDs = [self._IPR(images['C'], img) for img in self.answers]

            next_round_roster = []
            for index in range(6):
                if np.abs(iprAB * iprCDs[index]) < 2 or np.abs(iprAC * iprCDs[index]) < 2:
                    next_round_roster.append(index)
            if len(next_round_roster) > 0:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprAB - dprCDs[index]) for index in range(6)] ) + 1
            return choose

        elif self.problem.problemType == '3x3':
            # Calculate the metrics
            # A B C
            # D E F
            # G H I

            # Horizontal: GH -> HI
            # Vertical: CF -> FI
            # Diagnal: AE -> EI

            # 1. DPR - Dark Pixel Ratio
            # 2. IPR - Intersection Pixel Ratio
            # 3. Filter with IPR threshold
            # 4. define scoring
            ## the candidate image with the smallest abs dpr has the highest dpr score
            dprGH = self._DPR(images['G'], images['H'])
            iprGH = self._IPR(images['G'], images['H'])
            dprCF = self._DPR(images['C'], images['F'])
            iprCF = self._IPR(images['C'], images['F'])
            dprAE = self._DPR(images['A'], images['E'])
            iprAE = self._IPR(images['A'], images['E'])
            dprHIs = [self._DPR(images['H'], img) for img in self.answers]
            iprHIs = [self._IPR(images['H'], img) for img in self.answers]
            dprFIs = [self._DPR(images['F'], img) for img in self.answers]
            iprFIs = [self._IPR(images['F'], img) for img in self.answers]
            dprEIs = [self._DPR(images['E'], img) for img in self.answers]
            iprEIs = [self._IPR(images['E'], img) for img in self.answers]

            next_round_roster = []
            for index in range(8): # 3x3
                if np.abs(iprGH * iprHIs[index]) < 2:
                    next_round_roster.append((index, 'GH'))
                elif np.abs(iprCF * iprFIs[index]) < 2:
                    next_round_roster.append((index, 'CF'))
                elif np.abs(iprAE * iprEIs[index]) < 2:
                    next_round_roster.append((index, 'AE'))

            if len(next_round_roster) > 0:
                for index, direction in next_round_roster:
                    if direction == 'GH':
                        choose = np.argmin( [np.abs(dprGH - dprHIs[i]) for i, _ in next_round_roster] ) + 1
                    elif direction == 'CF':
                        choose = np.argmin( [np.abs(dprCF - dprFIs[i]) for i, _ in next_round_roster] ) + 1
                    elif direction == 'AE':
                        choose = np.argmin( [np.abs(dprAE - dprEIs[i]) for i, _ in next_round_roster] ) + 1
            else:
                choose = np.argmin( [np.abs(dprGH - dprHIs[index]) for index in range(8)] ) + 1

            #
            # print("%s: Best match %s" % (problem.name, choose))
            # print("DPR %s: %s" % (dprBC, dprHIs)) if DEBUG else None
            # print("IPR %s: %s" % (iprBC, iprHIs)) if DEBUG else None
            # print("DPR scores: %s" % scores) if DEBUG else None
            # # print("IPR scores: %s" % iprscores) if DEBUG else None
            # # print("TTL scores: %s" % totalsocres) if DEBUG else None
            # print("\n")
            return choose
    def _DPR(self, image1, image2):
        # Calculate the Dark Pixel Ratio
        dpr1 = np.sum(image1) / np.size(image1)
        dpr2 = np.sum(image2) / np.size(image2)
        return dpr1 - dpr2

    def _IPR(self, image1, image2):
        # Calculate the Intersection Pixel Ratio
        intersection = cv2.bitwise_or(image1, image2)
        ipr1 = np.sum(intersection) / np.sum(image1)
        ipr2 = np.sum(intersection) / np.sum(image2)
        return ipr1 - ipr2
    def Solve_Others(self, problem):
        # get the ideal image
        goalImages = self.getIdealImage(problem)
        # find the best match
        best_matches = self.find_best_match(problem, goalImages)
        return best_matches[0]

    # def Solve_DE(self, problem):
    #     self.problem = problem
    #
    #     # Image processing
    #     images = self.dataset_from_problem(problem)
    #
    #     self.answers = [images['1'], images['2'], images['3'], images['4'], images['5'], images['6']]
    #     if problem.problemType == "3x3":
    #         self.answers.append(images['7'])
    #         self.answers.append(images['8'])
    #
    #
    #     if self.problem.problemSetName in {"Basic Problems D", "Test Problems D", "Challenge Problems D", "Raven's Problems D"}:
    #         diff_bc = cv2.bitwise_xor(images['B'], images['C'])
    #         common_gh = cv2.bitwise_or(images['G'], images['H'])
    #         ans_diff = [cv2.bitwise_xor(images['H'], ans) for ans in self.answers]
    #         ans_common = [cv2.bitwise_or(images['H'], ans) for ans in self.answers]
    #
    #         threshold_max = np.sum(diff_bc) * 1.2
    #         threshold_min = np.sum(diff_bc) * 0.8
    #         threshold_list = [common for (common, diff) in zip(ans_common, ans_diff) if threshold_min <= np.sum(diff) <= threshold_max]
    #
    #         # find the answer with the closest common
    #         closest_index = np.argmin( [np.sum( _common - common_gh ) for _common in ans_common] )
    #         return closest_index+1
    #     elif self.problem.problemSetName in {"Basic Problems E", "Test Problems E", "Challenge Problems E", "Raven's Problems E"}:
    #         bitwise_and_gh = cv2.bitwise_and(images['G'], images['H'])
    #         ans_and = [cv2.bitwise_and(bitwise_and_gh, ans) for ans in self.answers]
    #         closest_index = np.argmin( [np.sum(_and - bitwise_and_gh) for _and in ans_and] )
    #         return closest_index+1
    #     return 1




    def calculate_bitwise(self, image1, image2):
        bitwise_or_h = cv2.bitwise_or(image1, image2)
        bitwise_xor_h = cv2.bitwise_xor(image1, image2)
        bitwise_xor_h_i = cv2.bitwise_not(bitwise_xor_h)
        bitwise_and_h = cv2.bitwise_and(image1, image2)
        ###
        cv2.imshow("bitwise_or_h", bitwise_or_h)
        cv2.waitKey(0)
        cv2.imshow("bitwise_xor_h", bitwise_xor_h)
        cv2.waitKey(0)
        cv2.imshow("bitwise_xor_h_i", bitwise_xor_h_i)
        cv2.waitKey(0)
        cv2.imshow("bitwise_and_h", bitwise_and_h)
        cv2.waitKey(0)


    def dataset_from_problem(self, problem):
        if problem.problemType == '2x2':
            A = cv2.imread(problem.figures["A"].visualFilename, cv2.IMREAD_GRAYSCALE)
            B = cv2.imread(problem.figures["B"].visualFilename, cv2.IMREAD_GRAYSCALE)
            C = cv2.imread(problem.figures["C"].visualFilename, cv2.IMREAD_GRAYSCALE)
            one = cv2.imread(problem.figures["1"].visualFilename, cv2.IMREAD_GRAYSCALE)
            two = cv2.imread(problem.figures["2"].visualFilename, cv2.IMREAD_GRAYSCALE)
            three = cv2.imread(problem.figures["3"].visualFilename, cv2.IMREAD_GRAYSCALE)
            four = cv2.imread(problem.figures["4"].visualFilename, cv2.IMREAD_GRAYSCALE)
            five = cv2.imread(problem.figures["5"].visualFilename, cv2.IMREAD_GRAYSCALE)
            six = cv2.imread(problem.figures["6"].visualFilename, cv2.IMREAD_GRAYSCALE)
            return {'problem_name': problem.name,
                    'A': A, 'B': B, 'C': C,
                    '1': one, '2': two, '3': three, '4': four, '5': five, '6': six}

        elif problem.problemType == '3x3':
            A = cv2.imread(problem.figures["A"].visualFilename, cv2.IMREAD_GRAYSCALE)
            B = cv2.imread(problem.figures["B"].visualFilename, cv2.IMREAD_GRAYSCALE)
            C = cv2.imread(problem.figures["C"].visualFilename, cv2.IMREAD_GRAYSCALE)
            D = cv2.imread(problem.figures["D"].visualFilename, cv2.IMREAD_GRAYSCALE)
            E = cv2.imread(problem.figures["E"].visualFilename, cv2.IMREAD_GRAYSCALE)
            F = cv2.imread(problem.figures["F"].visualFilename, cv2.IMREAD_GRAYSCALE)
            G = cv2.imread(problem.figures["G"].visualFilename, cv2.IMREAD_GRAYSCALE)
            H = cv2.imread(problem.figures["H"].visualFilename, cv2.IMREAD_GRAYSCALE)
            one = cv2.imread(problem.figures["1"].visualFilename, cv2.IMREAD_GRAYSCALE)
            two = cv2.imread(problem.figures["2"].visualFilename, cv2.IMREAD_GRAYSCALE)
            three = cv2.imread(problem.figures["3"].visualFilename, cv2.IMREAD_GRAYSCALE)
            four = cv2.imread(problem.figures["4"].visualFilename, cv2.IMREAD_GRAYSCALE)
            five = cv2.imread(problem.figures["5"].visualFilename, cv2.IMREAD_GRAYSCALE)
            six = cv2.imread(problem.figures["6"].visualFilename, cv2.IMREAD_GRAYSCALE)
            seven = cv2.imread(problem.figures["7"].visualFilename, cv2.IMREAD_GRAYSCALE)
            eight = cv2.imread(problem.figures["8"].visualFilename, cv2.IMREAD_GRAYSCALE)
            return {'problem_name': problem.name,
                    'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H,
                    '1': one, '2': two, '3': three, '4': four, '5': five, '6': six, '7': seven, '8': eight}
        else:
            print("Error: Problem {}: type not recognized".format(problem.name))
            return -1


    def detect_shape(self, image):
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []  # This will store the detected shapes
        for cnt in contours:
            # Approximate the contour to simplify it
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Example: classify shapes based on the number of vertices (simplified)
            if len(approx) == 3:
                shapes.append('triangle')
            elif len(approx) == 4:
                shapes.append('rectangle')
            # Add more shape classifications as needed
        return shapes




# if __name__ == "__main__":
    # agent = Agent()
    # problem_set = ProblemSet("Basic Problems C")
    # for i in range(1, 12):
    #     problem = problem_set.problems[i]
    #     ans = agent.Solve(problem)

    #     print("Problem %s: %s" % (problem.name, ans))

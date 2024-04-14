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
        return np.rot90(np_image, k = int(angle/90))

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
        if 'D' in problem.problemSetName:
            return self.Solve_DE(problem)
        elif 'E' in problem.problemSetName:
            return self.Solve_DE(problem)
        else:
            return self.Solve_Others(problem)

    def Solve_Others(self, problem):
        # get the ideal image
        goalImages = self.getIdealImage(problem)
        # find the best match
        best_matches = self.find_best_match(problem, goalImages)
        return best_matches[0]

    def Solve_DE(self, problem):
        self.problem = problem

        # Image processing
        self.A = Image.open(problem.figures['A'].visualFilename).convert('L')
        self.A = cv2.imread(problem.figures['A'].visualFilename, 0)
        _, self.A = cv2.threshold(self.A, 127, 255, cv2.THRESH_BINARY)
        self.B = Image.open(problem.figures['B'].visualFilename).convert('L')
        self.B = cv2.imread(problem.figures['B'].visualFilename, 0)
        _, self.B = cv2.threshold(self.B, 127, 255, cv2.THRESH_BINARY)
        self.C = Image.open(problem.figures['C'].visualFilename).convert('L')
        self.C = cv2.imread(problem.figures['C'].visualFilename, 0)
        _, self.C = cv2.threshold(self.C, 127, 255, cv2.THRESH_BINARY)
        if problem.problemType == "3x3":
            self.D = Image.open(problem.figures['D'].visualFilename).convert('L')
            self.D = cv2.imread(problem.figures['D'].visualFilename, 0)
            _, self.D = cv2.threshold(self.D, 127, 255, cv2.THRESH_BINARY)
            self.E = Image.open(problem.figures['E'].visualFilename).convert('L')
            self.E = cv2.imread(problem.figures['E'].visualFilename, 0)
            _, self.E = cv2.threshold(self.E, 127, 255, cv2.THRESH_BINARY)
            self.F = Image.open(problem.figures['F'].visualFilename).convert('L')
            self.F = cv2.imread(problem.figures['F'].visualFilename, 0)
            _, self.F = cv2.threshold(self.F, 127, 255, cv2.THRESH_BINARY)
            self.G = Image.open(problem.figures['G'].visualFilename).convert('L')
            self.G = cv2.imread(problem.figures['G'].visualFilename, 0)
            _, self.G = cv2.threshold(self.G, 127, 255, cv2.THRESH_BINARY)
            self.H = Image.open(problem.figures['H'].visualFilename).convert('L')
            self.H = cv2.imread(problem.figures['H'].visualFilename, 0)
            _, self.H = cv2.threshold(self.H, 127, 255, cv2.THRESH_BINARY)
        self.one = Image.open(problem.figures['1'].visualFilename).convert('L')
        self.one = cv2.imread(problem.figures['1'].visualFilename, 0)
        _, self.one = cv2.threshold(self.one, 127, 255, cv2.THRESH_BINARY)
        self.two = Image.open(problem.figures['2'].visualFilename).convert('L')
        self.two = cv2.imread(problem.figures['2'].visualFilename, 0)
        _, self.two = cv2.threshold(self.two, 127, 255, cv2.THRESH_BINARY)
        self.three = Image.open(problem.figures['3'].visualFilename).convert('L')
        self.three = cv2.imread(problem.figures['3'].visualFilename, 0)
        _, self.three = cv2.threshold(self.three, 127, 255, cv2.THRESH_BINARY)
        self.four = Image.open(problem.figures['4'].visualFilename).convert('L')
        self.four = cv2.imread(problem.figures['4'].visualFilename, 0)
        _, self.four = cv2.threshold(self.four, 127, 255, cv2.THRESH_BINARY)
        self.five = Image.open(problem.figures['5'].visualFilename).convert('L')
        self.five = cv2.imread(problem.figures['5'].visualFilename, 0)
        _, self.five = cv2.threshold(self.five, 127, 255, cv2.THRESH_BINARY)
        self.six = Image.open(problem.figures['6'].visualFilename).convert('L')
        self.six = cv2.imread(problem.figures['6'].visualFilename, 0)
        _, self.six = cv2.threshold(self.six, 127, 255, cv2.THRESH_BINARY)
        if problem.problemType == "3x3":
            self.seven = Image.open(problem.figures['7'].visualFilename).convert('L')
            self.seven = cv2.imread(problem.figures['7'].visualFilename, 0)
            _, self.seven = cv2.threshold(self.seven, 127, 255, cv2.THRESH_BINARY)
            self.eight = Image.open(problem.figures['8'].visualFilename).convert('L')
            self.eight = cv2.imread(problem.figures['8'].visualFilename, 0)
            _, self.eight = cv2.threshold(self.eight, 127, 255, cv2.THRESH_BINARY)

        self.answers = [self.one, self.two, self.three, self.four, self.five, self.six]
        if problem.problemType == "3x3":
            self.answers.append(self.seven)
            self.answers.append(self.eight)


        if self.problem.problemSetName in {"Basic Problems D", "Test Problems D", "Challenge Problems D", "Raven's Problems D"}:
            diff_bc = cv2.bitwise_xor(self.B, self.C)
            common_gh = cv2.bitwise_or(self.G, self.H)
            ans_diff = [cv2.bitwise_xor(self.H, ans) for ans in self.answers]
            ans_common = [cv2.bitwise_or(self.H, ans) for ans in self.answers]

            threshold_max = np.sum(diff_bc) * 1.5
            threshold_min = np.sum(diff_bc) * 0.6
            threshold_list = [common for (common, diff) in zip(ans_common, ans_diff) if threshold_min <= np.sum(diff) <= threshold_max]

            # find the answer with the closest common
            closest_index = np.argmin( [np.sum( _common - common_gh ) for _common in ans_common] )
            return closest_index+1
        elif self.problem.problemSetName in {"Basic Problems E", "Test Problems E", "Challenge Problems E", "Raven's Problems E"}:
            bitwise_and_gh = cv2.bitwise_and(self.G, self.H)
            ans_and = [cv2.bitwise_and(bitwise_and_gh, ans) for ans in self.answers]
            closest_index = np.argmin( [np.sum(_and - bitwise_and_gh) for _and in ans_and] )
            return closest_index+1
        return 1




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

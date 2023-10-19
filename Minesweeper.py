import math, heapq, numpy as np, random, copy, time
from pysat.solvers import Glucose3
from pysat.formula import CNF


class priorityQueue:
    def __init__(self) -> None:
        self.queue = []
        self.setLookUp = set()

    def push(self, priority, item):
        if not tuple(item) in self.setLookUp:
            heapq.heappush(self.queue, (priority, item))
            self.setLookUp.add(tuple(item))

    def pop(self):
        if(len(self.queue) == 0):
            return None
        item = heapq.heappop(self.queue)
        self.setLookUp.remove(tuple(item[1]))
        return item[1]

    def remove(self, priority, item):
        self.queue.remove((priority, item))
        self.setLookUp.remove(tuple(item))

    def sortQueue(self):
        self.queue.sort(key=lambda x: x[0])

    def is_in(self, item):
        return tuple(item) in self.setLookUp

    def isEmpty(self):
        return len(self.queue) == 0
    
    def __len__(self):
        return len(self.queue)



class Cell:
    def __init__(self,row , col, state) -> None:
        self.row = row
        self.col = col
        self.state = state

    def __str__(self) -> str:
        sign = ''
        if(self.state == False):
            sign = f'¬'
        return sign + f'P{self.row},{self.col}'


def A(n, k):
    return int(math.factorial(n)/math.factorial(n-k))

def hashFunc(src: str):
    num = 1
    for i in range(0, len(src)):
        num *= ord(src[i])
    return num

def atLeast(coords, k,exploreClauses, flag):
    result = []
    for i in range(0, len(coords)):
        j = i+1
        startPoint = i+1
        exploreLiterals = {}
        clause = []
        clause.append(Cell(coords[i][0],coords[i][1],flag))
        exploreLiterals[str(Cell(coords[i][0],coords[i][1],flag))] = 1
        while(j > i):
            j = j % len(coords)
            if(len(clause) >= k):
                key = ""
                for t in range(0, len(clause)):
                    key += str(clause[t])
                num = key.__hash__()
                if(exploreClauses.get(num,-1) == -1):
                    result.append(clause)
                    exploreClauses[num] = 1
                clause = []
                clause.append(Cell(coords[i][0],coords[i][1],flag))
                exploreLiterals = {}
                exploreLiterals[str(Cell(coords[i][0],coords[i][1],flag))] = 1
                if(len(coords) - startPoint >= k):
                    j = startPoint
                else:
                    break
                startPoint += 1
            else:
                key = str(Cell(coords[j][0],coords[j][1],flag))
                if(exploreLiterals.get(key,-1) == -1):
                    exploreLiterals[key] = 1
                    clause.append(Cell(coords[j][0],coords[j][1],flag))
            j += 1
    return result

def generatorClause(coords, valueNum,exploreSet):
    result = []
    uNumLiteral = valueNum + 1
    lNumLiteral = len(coords) - valueNum + 1
    if(uNumLiteral > len(coords) or lNumLiteral > len(coords)):
        return atLeast(coords,1,exploreSet,True)
    if(valueNum == 0):
        return atLeast(coords,1,exploreSet,False)
    dummy = atLeast(coords,uNumLiteral,exploreSet,False)
    if(dummy != []):
        result.extend(dummy)
    dummy = atLeast(coords,lNumLiteral,exploreSet,True)
    if(dummy != []):
        result.extend(dummy)
    return result
    


def constraintXY(matrix, row, col, valueNum,exploreSet):
    coords = []
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            if(row + i >=0 and row + i < len(matrix) and col + j >= 0 and col + j < len(matrix[row+i]) and matrix[row + i][col+j] == 0):
                coords.append([row+i,col+j])
    return generatorClause(coords,valueNum,exploreSet)

def generatorCNF(matrix):
    result = []
    exploreSet = {}
    for i in range(0,len(matrix)):
        for j in range(0, len(matrix[i])):
            if(matrix[i][j] != 0):
                dummy = constraintXY(matrix,i,j,matrix[i][j],exploreSet)
                if(dummy != []):
                    result.extend(dummy)
    return result

def xy2Int(cell : Cell, maxCol):
    result = 1
    if(cell.state == False):
        result = -1
    result = result * ((cell.row)* maxCol + cell.col+1)
    return result

def Int2xy(coord,maxRow ,maxCol):
    state = True
    if(coord < 0):
        state = False
        coord *= -1
    row = int((coord -1 )/ (maxRow))
    col = int((coord-1) % (maxCol))
    return row,col, state

def formClauseForPySat(clauses: list, model, maxCol):
    for i in range(0, len(clauses)):
        clause = []
        for j in range(0,len(clauses[i])):
            clause.append(xy2Int(clauses[i][j],maxCol))
        model.append(clause)

def findCoords(x, y, matrix):
    coords = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if(i == 0 and j == 0):
                continue
            if(x + i >= 0 and x + i < len(matrix) and y + j >= 0 and y + j < len(matrix[x+i])):
                coords.append([x+i,y+j])
    return coords

def solveCNF(matrix):
    clauses = generatorCNF(matrix)
    KB = CNF()
    duplicateClause = {}
    formClauseForPySat(clauses,KB,len(matrix[0]))
    for i in  range(0, len(KB.clauses)):
        if(len(KB.clauses[i]) == 1):
            duplicateClause[KB.clauses[i][0]] = 1
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if(matrix[i][j] != 0):
                coords = findCoords(i,j,matrix)
                maxBombs = matrix[i][j]
                counter = 0
                for k in range(0, len(coords)):
                    counterExample = []
                    counterExample.append(xy2Int(Cell(coords[k][0],coords[k][1],False),len(matrix[i])))
                    solver = Glucose3(bootstrap_with=KB)
                    if(duplicateClause.get(counterExample[0]*-1, -1) == -1 and solver.solve(assumptions=counterExample) == False):
                        counterExample[0] *= -1
                        duplicateClause[counterExample[0]] = 1
                        KB.append(counterExample)
                        counter += 1
                    solver.delete()
                    if(counter >= maxBombs):
                        break
    replicateBoard = []
    for i in range(0, len(matrix)):
        temp = []
        for j in range(0,len(matrix[i])):
            temp.append(False)
        replicateBoard.append(temp)
    for i in range(0,len(KB.clauses)):
        if(len(KB.clauses[i]) == 1):
            row, col, state = Int2xy(KB.clauses[i][0],len(matrix),len(matrix[0]))
            replicateBoard[row][col] = state
    result = []
    for i in range(0, len(matrix)):
        tempStr = ""
        for j in range(0, len(matrix[i])):
            if(replicateBoard[i][j] == False or matrix[i][j] != 0):
                tempStr += str(matrix[i][j])
            else:
                tempStr += "X"
            if(j + 1 < len(matrix[i])):
                tempStr += ", "
        result.append(tempStr) 
    return result
            
def booleanLogic(dictCell, str):
    if str[0] == '¬':
        return not dictCell[str[1:]]
    else:   
        return dictCell[str]

def check_one_clause(clause):
    if clause[0][0] == '¬':
        return False
    return True
def checkClause(clause,dictCell):
    logic=False
    list_Clause = clause.split(' ')
    for c in list_Clause:
        if c in dictCell:
            if(c[0] == '¬'):
                logic = logic or (not booleanLogic(dictCell,c))
                if logic:
                    return True
            else:
                logic=logic or booleanLogic(dictCell,c)
                if logic:
                    return True
        else:
            if(c[0] == '¬'):
                logic = logic or (not booleanLogic(dictCell,c))
                if logic:
                    return True
            else:
                logic=logic or booleanLogic(dictCell,c)
                if logic:
                    return True
    return logic
# Brute force thay cho bản cũ
def checkCNF_Brute_force(listTemp,dictCell):
    keys = list(dictCell.keys())
    num_keys = len(keys)
    num_states = 2 ** num_keys
    for state in range(num_states):
        for i in range(num_keys):
            dictCell[keys[i]] = bool((state >> i) & 1)
            check=True
            for clause in listTemp:
                if(not checkClause(clause,dictCell)):
                    check=False
                    break
            if(check):
                return True
    return False

# Backtracking thay thế cho bản cũ
def checkCNF_backtracking(listtemp, assignments, index=0):
    if index == len(listtemp):
        return is_satisfied(listtemp, assignments)
    
    clause = listtemp[index]
    list_clause = clause.split(' ')
    
    for literal in list_clause:
        if literal[0] == '¬':
            var = literal[1:]
            value = False
        else:
            var = literal
            value = True
        
        if var not in assignments:
            assignments[var] = value
            
            if is_satisfied(listtemp, assignments):
                if checkCNF_backtracking(listtemp, assignments, index + 1):
                    return True
            
            del assignments[var]
        else:
            if assignments[var] != value:
                return checkCNF_backtracking(listtemp, assignments, index + 1)
            
    return checkCNF_backtracking(listtemp, assignments, index + 1)

def is_satisfied(listtemp, assignments):
    for clause in listtemp:
        clause_satisfied = False
        literals = clause.split(' ')
        
        for literal in literals:
            if literal[0] == '¬':
                var = literal[1:]
                v_literal = False
            else:
                var = literal
                v_literal = True
            
            value = assignments.get(var)
            
            if value is None:
                assignments[var] = v_literal
                clause_satisfied = True
                break
            elif value == v_literal:
                clause_satisfied = True
                break
        
        if not clause_satisfied:
            return False
    
    return True

def change_matrix(matrix,dicCell):
    for c in dictCell:
        if dictCell[c]:
            x,y = c.split(',')
            matrix[int(x)][int(y)] = 'X'
#thêm hàm thay đổi ma trận
def change_matrix_a(matrix,assignments):
    for a in assignments:
        if assignments[a]:
            a=a[1:]
            x,y = a.split(',')
            matrix[int(x)][int(y)] = 'X'



def update8Square(x, y, matrix, maxLabel):
    counter =0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if(i == 0 and j == 0):
                continue
            if(x + i >= 0 and x + i < len(matrix) and y + j >= 0 and y + j < len(matrix[x+i])):
                matrix[x+i][y+j] += 1
                counter += 1
            if(counter >= maxLabel):
                return
    return

def matrixBombGenerator(maxRow, maxCol):
    matrix = []
    bombMax = int(maxCol*maxRow*10/100)
    counter = 0
    for i in range(0, maxRow):
        temp = []
        for j in range(0, maxCol):
            flag = random.randint(0,100)
            if(flag <= 5 and counter < bombMax):
                temp.append(-1)
                counter += 1
            else:
                temp.append(0)
        matrix.append(temp)
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if(matrix[i][j] == -1):
                matrix[i][j] = 0
                labels = random.randint(1,8)
                update8Square(i,j,matrix,labels)
    return matrix

def convertLiteralToInt(clause: Cell, rowLength):
    x, y, state = clause.row, clause.col, clause.state
    return (x*rowLength + y + 1) if state == True else -(x*rowLength + y + 1)


def formStrKB(KB, matrix):
    result = priorityQueue()
    for i in range(0, len(KB)):
        clause = []
        for j in range(0, len(KB[i])):
            clause.append(convertLiteralToInt(KB[i][j], len(matrix[0])))
            clause.sort()
        result.push(len(clause), clause)
    return result
        
def resolve(clause1, clause2):
    if (clause1 == clause2):
        return [0]
    if (clause1[0] == -clause2[0] and len(clause1) == 1 and len(clause2) == 1):
        return []
    result = []
    result.extend(clause1)
    result.extend(clause2)
    check = result.copy()
    temp = []
    flag = False
    resolved = False
    for i in range(0, len(clause1)):
        for j in range(0, len(clause2)):
            if (clause1[i] == -clause2[j]):
                result.remove(clause1[i])
                result.remove(clause2[j])
                if (result in temp):
                    result.append(clause1[i])
                    result.append(clause2[j])
                    continue
                temp.append(result.copy())
                result.append(clause1[i])
                result.append(clause2[j])
    if (result == check and temp == []):
        return [0]
    for i in range(0, len(temp)):
        temp[i] = sorted(set(temp[i]))
    return temp

def resolutionRefutation(KB, matrix, indexI, indexJ):
    clauses = copy.deepcopy(KB)
    clauses.push(1, [convertLiteralToInt(Cell(indexI, indexJ, False), len(matrix[0]))])
    while (True):
        temp = []
        clauses.sortQueue()
        iterator = copy.deepcopy(clauses)
        while (len(iterator) != 0):
            iteratorClause = iterator.pop()
            iterations = len(clauses)
            startIndex = clauses.queue.index((len(iteratorClause), iteratorClause))
            for j in range(startIndex, iterations):
                dummy = resolve(iteratorClause, clauses.queue[j][1])
                if (dummy == []):
                    return True
                elif (dummy == [0]):
                    continue
                else:
                    for t in range(len(dummy)):
                        if (clauses.is_in(dummy[t]) == False):
                            temp.append(dummy[t])
        if (temp == []):
            return False
        for i in range(0, len(temp)):
            clauses.push(len(temp[i]), temp[i])

def HeuristicResolution(matrix):
    resultMatrix = matrix.copy()
    KB = formStrKB(generatorCNF(matrix), matrix)
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if(matrix[i][j] == 0):             
                if (resolutionRefutation(KB,matrix, i,j) == True):
                    #print("There is a mine at ", i, " ", j)
                    resultMatrix[i][j] = 'X'
                else:
                    #print("There is no mine at ", i, " ", j)
                    resultMatrix[i][j] = '0'
            else:
                resultMatrix[i][j] = str(matrix[i][j])
    return resultMatrix


def preprocessing(matrix):
    listCell = set()
    temp = generatorCNF(matrix)
    matrix_b=matrix.copy()
    listtemp = []
    for i in range(0, len(temp)):
        s=""
        for j in range(0, len(temp[i])):
            if(j == 0):
                cell= str(temp[i][j])
                if cell[0] == '¬':
                    listCell.add(cell[1:]) 
                else:
                    listCell.add(cell)
                s += cell
            else:
                cell= str(temp[i][j])
                if cell[0] == '¬':
                    listCell.add(cell[1:]) 
                else:
                    listCell.add(cell)
                s +=' '+ cell
        listtemp.append(s)
    dictCell = dict.fromkeys(listCell, False)
    return listtemp,dictCell

if __name__ == '__main__':
    filename = "input.txt" # input file name
    print("Welcome to Minesweeper solver")
    print("Here are some guidelines: ")
    print("1. The input file must be in the same directory as the program and the program will read the file automatically")
    print("2. The input file's name is: " + str(filename))
    print("3. If the resulting matrix is the same as the input matrix, then there is no solution")
    print("4. The resulting matrix will be printed on screen and not exported to any files")
    print("5. The Pysat solver option is the most consistent to run with any test cases")
    print("______________________________________________________________________________________")
    input("Press Enter to continue...")
    file = open(filename, "r")
    matrix = []
    for line in file:
        temp = line.split(",")
        temp[len(temp)-1] = temp[len(temp)-1].replace("\n", "")
        matrix.append(temp)
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            matrix[i][j] = int(matrix[i][j])
    print("1. Brute force")
    print("2. Backtracking")
    print("3. Pysat solver")
    print("4. Search problem")
    mode = input("Choose the test algorithm: ")
    if(mode == "1"):
        print("Brute force")
        listtemp,dictCell = preprocessing(matrix)
        temp = matrix.copy()
        t_start = time.perf_counter()
        if checkCNF_Brute_force(listtemp,dictCell):
            change_matrix_a(temp,dictCell)
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if (j == len(temp[i]) - 1):
                        print(temp[i][j])
                    else:
                        print(temp[i][j], end=", ")
        else:
            print("No solution")
        t_end = time.perf_counter()
        print("Time taken (in seconds): ", t_end - t_start)
    elif(mode == "2"):
        assignments = {}
        listtemp,dictCell = preprocessing(matrix)
        temp = matrix.copy()
        t1_start = time.perf_counter()
        if checkCNF_backtracking(listtemp, assignments):
            change_matrix_a(temp,assignments)
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if (j == len(temp[i]) - 1):
                        print(temp[i][j])
                    else:
                        print(temp[i][j], end=", ")
        else:
            print("No solution")
        t1_end = time.perf_counter()
        print("Time taken (in seconds): ", t1_end - t1_start)
    elif(mode == "3"):
        t1_start = time.perf_counter()
        for i in solveCNF(matrix):
            print(i)
        t1_end = time.perf_counter()
        print("Time taken (in seconds): ", t1_end - t1_start)
    elif(mode == "4"):
        t1_start = time.perf_counter()
        HeuristicResolution(matrix)
        t1_end = time.perf_counter()
        print("Time taken (in seconds): ", t1_end - t1_start)
    else:
        print("Invalid input")
    file.close()
    print("Program terminated")
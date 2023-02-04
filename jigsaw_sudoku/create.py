#!/usr/bin/env python3
#
# Some portions from: https://www.ocf.berkeley.edu/~arel/sudoku/main.html

import argparse
import numpy as np
import numpy.random as npr
import torch

from tqdm import tqdm

import os, sys
import shutil
import itertools

import random, copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from PIL import Image
import glob
import random

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

# map of (i, j) to partition 
GEOMETRIC_VARIANTS= [
    {
        # standard square
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 1,
        (0, 3): 1,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 1,
        (2, 0): 2,
        (2, 1): 2,
        (2, 2): 3,
        (2, 3): 3,
        (3, 0): 2,
        (3, 1): 2,
        (3, 2): 3,
        (3, 3): 3,
    },
    {
        # variant 1
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 1,
        (0, 3): 1,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 0,
        (1, 3): 1,
        (2, 0): 2,
        (2, 1): 3,
        (2, 2): 3,
        (2, 3): 3,
        (3, 0): 2,
        (3, 1): 2,
        (3, 2): 2,
        (3, 3): 3,
    },
    {
        # variant 2
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 1,
        (1, 0): 2,
        (1, 1): 2,
        (1, 2): 0,
        (1, 3): 1,
        (2, 0): 2,
        (2, 1): 3,
        (2, 2): 1,
        (2, 3): 1,
        (3, 0): 2,
        (3, 1): 3,
        (3, 2): 3,
        (3, 3): 3,
    }
]

HUES = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boardSz', type=int, default=2)
    parser.add_argument('--nSamples', type=int, default=100)
    parser.add_argument('--data', type=str, default='data_test')
    parser.add_argument('--dataGen', action='store_true')
    parser.add_argument('--nHoles', type=int, default=1)
    args = parser.parse_args()

    assert(args.boardSz == 2)

    npr.seed(0)
    if (args.dataGen):
        gen_geometric_variant(100)

    save = os.path.join(args.data, str(args.boardSz))
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save)
    os.makedirs(os.path.join(save, "images"))
    os.makedirs(os.path.join(save, "solution"))

    Y = []
    for i in tqdm(range(args.nSamples)):
        Xi, Yi = sample_geom(args, i)
        Y.append(Yi)

    Y = np.array(Y)

    loc = "labels.pt"
    arr = Y
    fname = os.path.join(save, loc)
    with open(fname, 'wb') as f:
        torch.save(torch.Tensor(arr), f)
    print("Created {}".format(fname))


def gen_geometric_variant(n, boardSz=2, FAIL_LIMIT=100):
    '''
    generates n new geometric variants
    no guarantees of uniqueness
    '''
    num_created = 0
    while num_created < n:
        geometric_variant = {}
        fail_count = 0
        all_cells = [(i,j) for i, j in itertools.product(range(boardSz * boardSz), range(boardSz * boardSz))]
        current_all_cells = list(all_cells)
        for partition in range(boardSz * boardSz):
            partition_all_cells = list(current_all_cells)
            partition_elements = []
            # randomly select one cell for the partition
            while fail_count < FAIL_LIMIT and len(partition_elements) != boardSz * boardSz:
                partition_element = random.choice(partition_all_cells)
                partition_all_cells.remove(partition_element)
                partition_elements.append(partition_element)
                while len(partition_elements) < boardSz * boardSz:
                    curr_partition_element = partition_elements[-1]
                    curr_i, curr_j = curr_partition_element
                    # find cells adjacent that are still there
                    candidate_elements = []

                    for i in range(curr_i - 1, curr_i + 2):
                        for j in range(curr_j - 1, curr_j + 2):
                            if (i,j)  in partition_all_cells and \
                                abs(curr_i - i) + abs(curr_j - j) != 2 and \
                                abs(curr_i - i) + abs(curr_j - j) != 0:
                                candidate_elements.append((i,j))
                    
                    if len(candidate_elements) == 0:
                        fail_count += 1
                        partition_all_cells.append(curr_partition_element)
                        partition_elements.remove(curr_partition_element)
                        break 
                    else:
                        # randomly pick one
                        new_partition_element = random.choice(candidate_elements)
                        partition_all_cells.remove(new_partition_element)
                        partition_elements.append(new_partition_element)
                        
            
            if len(partition_elements) == boardSz * boardSz:
                for pe in partition_elements:
                    geometric_variant[pe] = partition
                    current_all_cells.remove(pe)
            else:
                # failed to find a partition, just redo the while board
                geometric_variant = {}
                fail_count = 0
                break
        if len(current_all_cells) == 0:
            # no cells left
            GEOMETRIC_VARIANTS.append(geometric_variant)
            geometric_variant = {}
            fail_count = 0
            num_created += 1


def lookup_image(num):
    fp = f'mnist_png/testing/{num}'
    l = glob.glob(f"{fp}/*.png")
    img_p = random.choice(l)
    return Image.open(img_p).convert('RGB')


def image_grid_geom(imgs, rows, cols, geom):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i in range(rows):
        for j in range(cols):
            layer = Image.new('RGB', imgs[i * cols + j].size, HUES[GEOMETRIC_VARIANTS[geom][(i, j)]]) # "hue" selection is done by choosing a color...
            output = Image.blend(imgs[i * cols + j], layer, 0.5)
            grid.paste(output, box=(i*w, j*h))
    return grid


def create_img_puzzle_geom(puzzle, args, fname, geom):
    boardSz = args.boardSz
    Nsq = boardSz * boardSz
    imgs = []
    # lookup image in mnist_png for each entry
    for i in range(Nsq):
        for j in range(Nsq):
            num = puzzle[i][j]
            img = lookup_image(num)
            imgs.append(img) 
    
    grid = image_grid_geom(imgs, Nsq, Nsq, geom)
    grid.save(f"{fname}.png", "PNG")


# geom only has one blank
def sample_geom(args, i, n_blanks=1):
    solution, geom = construct_geometric_solution(args.boardSz)
    Nsq = args.boardSz*args.boardSz
    board, lcells = pluck_geom(copy.deepcopy(solution), (Nsq * Nsq) - args.nHoles, geom)
    create_img_puzzle_geom(board, args, f'{args.data}/2/images/{i}', geom)
    create_img_puzzle_geom(solution, args, f'{args.data}/2/solution/{i}', geom)
    solution = toOneHot(solution)
    
    return board, solution


def toOneHot(X):
    X = np.array(X)
    Nsq = X.shape[0]
    Y = np.zeros((Nsq, Nsq, Nsq))
    for i in range(1,Nsq+1):
        Y[:,:,i-1][X == i] = 1.0
    return Y


def construct_geometric_solution(N):
    """
    Randomly arrange numbers in a grid while making all rows, columns and
    geometric shapes (sub-grids) contain the numbers 1 through Nsq.

    For example, "sample" (above) could be the output of this function. """
    Nsq = N*N
    variant = random.randint(0, len(GEOMETRIC_VARIANTS) - 1)
    while True:
        try:
            puzzle  = [[0]*Nsq for i in range(Nsq)] # start with blank puzzle
            rows    = [set(range(1,Nsq+1)) for i in range(Nsq)] # set of available
            columns = [set(range(1,Nsq+1)) for i in range(Nsq)] #   numbers for each
            geoms = [set(range(1,Nsq+1)) for i in range(Nsq)] #   row, column and geoms
            for i in range(Nsq):
                for j in range(Nsq):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(
                        geoms[GEOMETRIC_VARIANTS[variant][(i,j)]])
                    choice  = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    geoms[GEOMETRIC_VARIANTS[variant][(i,j)]].discard(choice)

            # success! every cell is filled.
            return puzzle, variant

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass


def get_cell_partition(geom, c_row, c_col):
    '''
    gets the partition which cell at (c_row, c_col) belongs to'''
    shape = GEOMETRIC_VARIANTS[geom]
    partition = shape[(c_row, c_col)]
    return partition


def get_geom_cells(geom, c_row, c_col):
    '''
    get all cells (i, j) which belong to the same geometric partition as cell (c_row, c_col)'''
    shape = GEOMETRIC_VARIANTS[geom]
    partition = shape[(c_row, c_col)]
    cells_in_part = []
    for k, v in shape.items():
        if v == partition:
            cells_in_part.append(k)

    return cells_in_part


# we only discard one cell
def pluck_geom(puzzle, nKeep, geom):
    Nsq = len(puzzle)
    N = int(np.sqrt(Nsq))

    def canBeA(puz, i, j, c):
        """
        Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
        in cell "c"? """
        v = puz[c//Nsq][c%Nsq]
        c_row = c//Nsq
        c_col = c%Nsq
        if puz[i][j] == v: return True
        if puz[i][j] in range(1,Nsq+1): return False

        for m in range(Nsq): # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m==c//Nsq and j==c%Nsq) and puz[m][j] == v: return False
            if not (i==c//Nsq and m==c%Nsq) and puz[i][m] == v: return False
            geom_cells = get_geom_cells(geom, c_row, c_col)
            if not (geom_cells[m] == (c_row, c_col)) and puz[geom_cells[m][0]][geom_cells[m][1]] == v:
                return False
        return True

    """
    starts with a set of all N^4 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells     = set(range(Nsq*Nsq))
    cellsleft = cells.copy()
    cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
    cellsleft.discard(cell) # record that we are trying this cell

    while len(cells) > nKeep and len(cellsleft):
        cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
        cellsleft.discard(cell) # record that we are trying this cell
        c_row = cell//Nsq
        c_col = cell%Nsq
        # row, col and shape record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        row = col = shape = False

        for i in range(Nsq):
            geom_cells = get_geom_cells(geom, c_row, c_col)
            if i != cell//Nsq:
                if canBeA(puzzle, i, cell%Nsq, cell): row = True
            if i != cell%Nsq:
                if canBeA(puzzle, cell//Nsq, i, cell): col = True
            if not (geom_cells[i] != (c_row, c_col)):
                if canBeA(puzzle, geom_cells[i][0], geom_cells[i][1], cell): shape = True

        if row and col and shape:
            continue # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            puzzle[cell//Nsq][cell%Nsq] = 0 # 0 denotes a blank cell
            cells.discard(cell) # remove from the set of visible cells (pluck it)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))


if __name__=='__main__':
    main()

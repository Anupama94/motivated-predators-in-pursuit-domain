import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import time
import pandas as pd
from pathlib import Path
from numba import njit, jit
from numba.typed import List, Dict
from numba import types, typeof, typed
import numpy as np
import concurrent
from concurrent.futures import ThreadPoolExecutor
from numba import int32, float32, deferred_type

from arena import Env
from argumentList import ArgumentList
from grid import Grid
from game import Game
from movement import predatorNextMove
from modifiedIncentiveFunctions import getModifiedPowerMotivationTendency, getModifiedAffiliationMotivationTendency,\
    calculateModifiedGlobalIncentive, getModifiedAchievementMotivationTendency
from modifiedUtils import configurePreys, configurePredators, assignPosition, rand_choice_nb,\
    modified_calculateLocalIncentivesForEachGoal, modified_localEfficiencyIncentive, getClosestTargetCell, formatFolderName, \
    map_difficulty_level




def parse_args():
    parser = argparse.ArgumentParser("Motivated rule based environments for pursuit-evasion environments")
    # Environment
    parser.add_argument("--predator-count", type=int, default=12, help="number of predators onn the arena")
    parser.add_argument("--prey-count", type=int, default=12, help="maximum number of preys")
    parser.add_argument("--regiment", type=str, default="MIXED", help="composition of motive profiles in the team")
    parser.add_argument("--num-games", type=int, default=100, help="number of games to run to get statisticsm")
    parser.add_argument("--render-frequency_easy", type=float, default=0.5, help="how often should the UI be rendered")
    parser.add_argument("--grid-size", type=int, default=16, help="value of n in a nxn grid")
    parser.add_argument("--assign-agent-locations", type=str, default="RANDOM", help="assign locations to the agents in the arena manually or randomly")
    parser.add_argument("--stationary-preys", type=int, default=1, help="should the preys move or not")
    parser.add_argument("--prey-capture-condition", type=int, default=4, help="number of predators required to cpature a prey")
    parser.add_argument("--motive-profile-ratio", nargs='+', type=int, default=[12, 0, 0], help="ratio of predator count belonging to different motives")
    parser.add_argument("--difficulty-level", type=int, default=0,
                        help="distribution of significance level of predators in the arena")

    parser.add_argument("--predator-vision-range", type=int, default=15, help="vision range of predators")
    parser.add_argument("--predator-communication-range", type=int, default=15, help="communication range of predators")
    parser.add_argument("--enable-gui", type=bool, default=True, help="enable UI")



    return parser.parse_args()

@njit(nogil=True)
def startGame(arglist):
    gameStats = np.full((arglist.prey_count + 43), -1.0)
    columns = np.arange(0, arglist.prey_count + 43)
    for i in range(1):
        grid = Grid(arglist.grid_size, arglist.grid_size)

        # env = Env(grid, arglist.enable_gui)
        arglist.motive_profile_ratio = np.array([arglist.aff_count, arglist.pow_count, arglist.ach_count])
        preys = configurePreys(arglist.prey_count, arglist.difficulty_level)
        predators = configurePredators(arglist.predator_count, arglist.motive_profile_ratio)

        occupiedPositions = np.zeros(shape=(arglist.predator_count+arglist.prey_count, 2))
        if arglist.assign_agent_locations == "RANDOM":

            # set random locations for preys
            counter = 0
            for agent in preys:
                row, col, occupiedPositions = assignPosition(occupiedPositions, arglist.grid_size, counter)
                grid.occupyCellByPrey(row, col, agent)
                counter += 1

            # set random locations for predators
            for agent in predators:
                row, col, occupiedPositions = assignPosition(occupiedPositions, arglist.grid_size, counter)
                grid.occupyCellByPredator(row, col, agent)
                counter += 1


        steps = 0
        intrinsicReward = 1
        preyYield = 0
        yieldIn200Steps = 0
        tensionIn200Steps = 1
        tensionIn200Steps2 = 1
        yieldIn400Steps = 0
        tensionIn400Steps = 1
        tensionIn400Steps2 = 1
        yieldIn600Steps = 0
        tensionIn600Steps = 1
        tensionIn600Steps2 = 1
        yieldIn1000Steps = 0
        tensionIn1000Steps = 1
        tensionIn1000Steps2 = 1
        yieldIn1200Steps = 0
        tensionIn1200Steps = 1
        yieldIn1400Steps = 0
        tensionIn1400Steps = 1
        yieldIn1600Steps = 0
        tensionIn1600Steps = 1
        yieldIn1800Steps = 0
        tensionIn1800Steps = 1
        yieldIn800Steps = 0
        tensionIn800Steps = 1
        yieldIn500Steps = 0
        tensionIn500Steps = 1
        yieldIn2000Steps = 0
        tensionIn2000Steps = 1
        yieldperunittension = 1

        game = Game(grid, 4, List(preys), List(predators))

        while True:
            steps += 1

            for preyIdx in range(len(game.preys)):
                prey = game.preys[preyIdx]
                if prey.status == "ALIVE":
                    if arglist.stationaryPreys == 1:
                        unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(prey.currentPosition)

                        dists = np.zeros(shape=(len(unoccupiedNeighbourCells)))
                        for cellInd in range(len(unoccupiedNeighbourCells)):
                            cell = unoccupiedNeighbourCells[cellInd]
                            totalDist = 0
                            for pred in game.predators:
                                totalDist += grid.getManhattenDistance(cell, pred.currentPosition)
                            dists[cellInd] = totalDist


                        if len(unoccupiedNeighbourCells) > 0:
                            idx = dists.argmax()
                            cellWithMaxDist = unoccupiedNeighbourCells[idx]
                            shouldRest = rand_choice_nb(np.array([0, 1]), np.array([1-prey.stamina, prey.stamina]))
                            if shouldRest == 0:
                                grid.movePreyToNewCell(prey, cellWithMaxDist)



                    if game.isPreyCaptured(prey):
                        game.arena.markDeadPrey(prey)
                        gameStats[arglist.prey_count - game.numberOfPreysLeft] = steps
                        columns[arglist.prey_count - game.numberOfPreysLeft] = prey.id*-1000 + prey.sig
                        game.numberOfPreysLeft = game.numberOfPreysLeft - 1

                        # print("Prey ", prey.id, " was captured in ", steps)

                        preyYield += prey.sig

            if game.isGameOver() or steps >= 5000:
                finalCount = len(game.preys)
                gameStats[finalCount] = steps
                gameStats[finalCount+1] = intrinsicReward
                gameStats[finalCount+2] = (preyYield * 1000) / steps

                gameStats[finalCount + 34] = intrinsicReward / steps



                columns[finalCount] = 111.0 # step count
                columns[finalCount+1] = 222.0 # perceived_tension
                columns[finalCount+2] = 333.0 # yield

                columns[finalCount + 13] = 784.0

                columns[finalCount + 34] = 805.0

                print(i, " GAME OVER !!!!!", "STEPS", steps)
                break

            # shuffled indices for predators
            predatorOrder = np.arange(len(game.predators))
            np.random.shuffle(predatorOrder)
            for predatorOrderInd in range(arglist.predator_count):
                predator = game.predators[predatorOrder[predatorOrderInd]]

                localIncentives = []
                localEffIncentives = []
                globalIncentives = []
                for pIdx in range(len(game.preys)):
                    p = game.preys[pIdx]
                    if p.status == "ALIVE":
                        # calculating motivation incentives
                        localIncentives.append((p, modified_calculateLocalIncentivesForEachGoal(p,\
                                                                                                game.predators, grid,\
                                                                                                arglist.significance_weight,\
                                                                                                arglist.adjacent_weight,
                                                                                                arglist.local_aff_range)))

                        # caluclating local eff incentiives
                        localEffIncentives.append((p, modified_localEfficiencyIncentive(p, grid,
                                                                               predator, arglist.threshold_distance))) # ASK THARAKA

                        # find best prey to be targetted globally
                        modifiedGlobalIncentive = calculateModifiedGlobalIncentive(arglist.global_aff_range, p, game.predators, grid,
                                                                                   game.numberOfPreysLeft, predator)
                        globalIncentives.append((p, modifiedGlobalIncentive))

                if predator.policy == "power":
                    pdf = getModifiedPowerMotivationTendency(len(localIncentives))

                elif predator.policy == "achievement":
                    pdf = getModifiedAchievementMotivationTendency(len(localIncentives))

                else:
                    pdf = getModifiedAffiliationMotivationTendency(len(localIncentives))

                sortedLocalIncentives = sorted(localIncentives, key=lambda x: float(x[1]))
                localEfficiencyTendencyValues = [x[1] for x in localEffIncentives]
                globalTendencyValues = [x[1] for x in globalIncentives]

                maxOfLocalEfficiencyTendencies  = max(localEfficiencyTendencyValues)
                maxOfGlobalTendencies = max(globalTendencyValues)

                alpha = np.array([1, 2*maxOfLocalEfficiencyTendencies, 3*maxOfGlobalTendencies])

                winner = np.random.dirichlet(alpha, size=1)[0].argmax()

                selectedRandomNumber = rand_choice_nb(np.array([x for x in range(len(localIncentives))]), np.array(pdf))
                incentiveForPreferredGoal = sortedLocalIncentives[selectedRandomNumber][1]

                if winner == 0:
                    predator.targetPrey = sortedLocalIncentives[selectedRandomNumber][0]
                    incentiveForChosenGoal = incentiveForPreferredGoal
                elif winner == 1:
                    predator.targetPrey = localEffIncentives[np.array(localEfficiencyTendencyValues).argmax()][0]
                    intrinsicReward += 1
                    incentiveForChosenGoal = localIncentives[np.array(localEfficiencyTendencyValues).argmax()][1]
                else:
                    predator.targetPrey = globalIncentives[np.array(globalTendencyValues).argmax()][0]
                    intrinsicReward += 1
                    incentiveForChosenGoal = localIncentives[np.array(globalTendencyValues).argmax()][1]

                # calculate pressure / tension
                yieldperunittension += abs(incentiveForPreferredGoal - incentiveForChosenGoal)

                dist = grid.getManhattenDistance(predator.targetPrey.currentPosition, predator.currentPosition)

                predatorShouldRest = rand_choice_nb(np.asarray([0, 1]), np.asarray([0.98, 0.02]))
                if predatorShouldRest == 0:
                    if dist > 1:
                        predatorNextMove(predator, predator.targetPrey, grid)


            # if arglist.enable_gui == True:
            #     # time.sleep(0.2)
            #     env.update_grid()

    return (gameStats, columns)

if __name__ == '__main__':
    parsedArgs = parse_args()
    arglist = ArgumentList()
    arglist.predator_count = parsedArgs.predator_count
    arglist.prey_count = parsedArgs.prey_count
    arglist.regiment = parsedArgs.regiment
    arglist.difficulty_level = parsedArgs.difficulty_level
    arglist.assign_agent_locations = parsedArgs.assign_agent_locations
    arglist.aff_count = int(parsedArgs.motive_profile_ratio[0])
    arglist.pow_count = int(parsedArgs.motive_profile_ratio[1])
    arglist.ach_count = int(parsedArgs.motive_profile_ratio[2])

    startTime = time.time()
    futures = []
    stats = []
    allStats = []

    with ThreadPoolExecutor(25) as ex:
        for i in range(400):
            futures.append(ex.submit(startGame, arglist))
        for future in concurrent.futures.as_completed(futures):
            stats.append(future.result())

    for i in range(400):
        data = stats[i][0]
        columns = stats[i][1]
        df = pd.DataFrame(data=[data], columns=columns)
        allStats.append(df)
    parentFolderName = str(arglist.prey_count) + "_preys_" + map_difficulty_level(arglist.difficulty_level)
    folderName = arglist.regiment + "_" + formatFolderName(arglist.motive_profile_ratio)
    csvFileName = str(os.getpid()) + '.csv'
    Path(
        '../resultsCSV/' + parentFolderName + '/' + folderName).mkdir(
        parents=True,
        exist_ok=True)
    with open(
            '../resultsCSV/' + parentFolderName + '/' + folderName + '/' + csvFileName,
            'w') as f:
        pd.concat(allStats).to_csv(f, index=False)



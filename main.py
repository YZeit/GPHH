import streamlit as st
from functions import geneticprogamming
from functions import geneticprogamming_speedup
from functions import geneticprogamming_verification
from functions import start_new_project
from functions import compute_reference_points
from functions import initialization_table
from functions import calculate_ref_point_parameter
from functions import objective_ideal
from functions import objective_nadir
from functions import visualize_ref_points
from functions import verification
from functions import verification_verification
from functions import geneticprogamming_speedup_benchmark
from functions import create_statistics_visualization
from functions import create_tree_visualization
from functions import save_values
from functions import load_test_problem
from functions import create_schedule_visualization
from functions import calculate_weightings_ASF
from functions import geneticprogamming_simulation
from functions import create_relative_performance
from functions import dominates
from functions import keep_efficient
from functions import is_pareto_efficient_dumb
from functions import create_radar_chart
from functions import create_visualization_zoom
from functions import eliminate_solution
from functions import create_relative_performance_final
from functions import verification_final
from functions import create_tree_visualization_final_solution
from functions import create_evaluation_terminals
import pandas as pd
import openpyxl as op
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import random as random

st.write("""
# Interactive Decision Support Tool
This is an interactive Tool to help you to optimize your production schedule.
""")

terminals_table = pd.read_excel('evaluation_terminals.xlsx', sheet_name='Sheet6', header=0)
st.write(terminals_table)
names_terminals = terminals_table['Terminal']
terminals = terminals_table['Occurence']
st.write(names_terminals)
st.write(terminals)
fig = create_relative_performance(makespan=terminals, tardiness=terminals, waiting=terminals, x_labels=names_terminals)
st.pyplot(fig)

# set parameters for genetic programming
st.sidebar.header('Select the genetic programming parameter')
nb_generations = st.sidebar.slider('number of generations', min_value=1, max_value=100, value=50, step=1)
population_size = st.sidebar.slider('population size', min_value=1, max_value=500, value=200, step=1)
crossover_probability = st.sidebar.slider('crossover probability', min_value=0.01, max_value=0.99, value=0.90, step=0.01)
mutation_probability = st.sidebar.slider('mutation probability', min_value=0.01, max_value=0.99, value=0.10, step=0.01)
max_depth_crossover = st.sidebar.slider('maximum depth for crossover', min_value=1, max_value=20, value=17, step=1)
max_depth_mutation = st.sidebar.slider('maximum depth for mutation', min_value=1, max_value=20, value=7, step=1)

# set parameter for fitness evaluation
st.sidebar.header('Select the fitness evaluation parameter')
nb_simulations = st.sidebar.slider('number of simulation runs', min_value=1, max_value=100, value=10, step=1)

# Performance Names; here fixed values; later as an option to type in or select in app
performance_name = ["Makespan", "Total tardiness", "Total waiting time"]

st.header('Start a new project or load an existing project')
st.write('If you want to load an existing project, please type the name and press enter. '
         'Otherwise, you can create a new project. Please fill in the name and press the button.')
project_name = st.text_input("Project name", key="project name")
if st.button("Create a new project"):
    start_new_project(project_name, performance_name)

# try open the initialization data
initialization_table = initialization_table(project_name=project_name)

st.header('Step 1: Initialization Phase')
st.write("""In this step you will learn more about the optimization model and the objective space. Therefore, a set of non-dominated solutions 
         will be provided. These solutions are automatically calculated by the system based on uniformly distributed reference points on a hyper-plane between the ideal and
         nadir point. These solutions can be compared to each other or used for further explorations in the second step.""")

st.subheader('Approximate the ideal and nadir point')
st.write('If you have already calculated the ideal and nadir point, they will be shown below. Otherwise, you can recalculate the '
         'ideal and nadir point by clicking the button.')

if st.button("Approximate ideal and nadir point"):
    # compute an approximation of the ideal and nadir point for each objective (here: 3)
    number_objectives = 3
    for i in range(number_objectives):
        best_solution, best_fitness, nb_generation_best, avgFitnessValues_best, minFitnessValues_best, maxFitnessValues_best, stdFitnessValues_best = \
            geneticprogamming_simulation(performance_measure=i, ref_point=None, opt_type="best", project_name=project_name,
                          reference_point_input=[0,0,0], nb_generations=nb_generations, population_size=population_size,
                          crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                          max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                          nb_simulations=nb_simulations, lambda_performance=None, constraint_index=None, constraint_value=None)
        save_values(project_name=project_name, ref_point=None, performance_name=performance_name, performance_measure=i, opt_type='best',
                    best_fitness=best_fitness, reference_point_input=None, best_solution=best_solution, performance=[0,0,0])
        best_solution, best_fitness, nb_generation_worst, avgFitnessValues_worst, minFitnessValues_worst, maxFitnessValues_worst, stdFitnessValues_worst = \
            geneticprogamming_simulation(performance_measure=i, ref_point=None, opt_type="worst", project_name=project_name, reference_point_input=[0,0,0],
                          nb_generations=nb_generations, population_size=population_size,
                          crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                          max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                          nb_simulations=nb_simulations, lambda_performance=None, constraint_index=None, constraint_value=None)
        save_values(project_name=project_name, ref_point=None, performance_name=performance_name, performance_measure=i, opt_type='worst',
                    best_fitness=best_fitness, reference_point_input=None, best_solution=best_solution, performance=[0,0,0])
    st.stop()
st.write(initialization_table)

st.subheader('Set the initial Reference Points')
st.write("""The Reference Point shown below are automatically calculated by the system based on the approximated ideal and 
nadir point. The Reference Points are shown in the table as well as in the objective space. Please check the position of the hyper-plane and, if desired, adjust it by using the slider.""")

# select the position of the uniformly distributed reference points
choice_ref_point = st.select_slider('Select the position of the reference points', options=['ideal point', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'nadir point'], value=0.4)
parameter_ref_point = calculate_ref_point_parameter(choice_ref_point=choice_ref_point)

# load the ideal and nadir point data
objective_ideal = objective_ideal(initialization_table=initialization_table)
objective_nadir = objective_nadir(initialization_table=initialization_table)

# calculate the lambda weightings for the ASF
weightings_ASF = calculate_weightings_ASF(objective_ideal, objective_nadir)

# compute and show the reference points
reference_points, reference_points_new = compute_reference_points(objective_ideal=objective_ideal, objective_nadir=objective_nadir, parameter_ref_point=parameter_ref_point)

# visualize reference points
st.subheader('Reference Points (Objective Space)')
visualize_ref_points(reference_points=reference_points, reference_points_new=reference_points_new, objective_ideal=objective_ideal, objective_nadir=objective_nadir)

st.subheader('Calculate Solutions')
st.write("""Based on the previously presented reference points, the system is automatically calculating a set of 
non-dominated solutions. For each reference point, the achievement scalarizing function is applied which computes
 a non-dominated solution that minimizes the distance to the reference point. To compute the solutions, please press the button.""")

if st.button('Compute solutions'):
    for i in range(len(reference_points_new.T)):
        reference_point = [reference_points_new.iloc[0,i], reference_points_new.iloc[1, i], reference_points_new.iloc[2, i]]
        best_solution, best_fitness, nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues, stdFitnessValues = \
            geneticprogamming_simulation(performance_measure=1, ref_point=True, opt_type="best",
                                                project_name=project_name,
                                                reference_point_input=reference_point,
                                                nb_generations=nb_generations, population_size=population_size,
                                                crossover_probability=crossover_probability,
                                                mutation_probability=mutation_probability,
                                                max_depth_crossover=max_depth_crossover,
                                                max_depth_mutation=max_depth_mutation,
                                                nb_simulations=nb_simulations, lambda_performance=weightings_ASF, constraint_index=None, constraint_value=None)

        # verify the solution and evaluate the performance measures
        fitness, results, performance = verification(performance_measure=1, ref_point=True,
                                                     reference_point_input=reference_point,
                                                     best_func=best_solution, lambda_performance=weightings_ASF, nb_simulations=nb_simulations)
        # save the solution
        save_values(project_name=project_name, ref_point=True, performance_name=performance_name, performance_measure='1', opt_type='best',
                    best_fitness=best_fitness, reference_point_input=reference_point, best_solution=best_solution, performance=performance)

        # visualize GP statistics in form of a errorchart
        GP_statistics = create_statistics_visualization(nb_generation, avgFitnessValues, minFitnessValues,
                                                        maxFitnessValues)
        st.pyplot(GP_statistics)

        # create tree visualization of the best solution found
        image_tree = create_tree_visualization(best_solution)
        st.header('Primitive tree of the best solution')
        st.image(image_tree, use_column_width=True)

        # show the best solution found as a string representation
        st.header('Best dispatching rule found')
        st.write(str(best_solution))

        # show the fitness of the best solution
        st.header('Fitness of best solution')
        st.write(best_fitness)

        st.header('Performance measures of the best solution')
        for i in range(len(performance_name)):
            st.subheader(performance_name[i])
            st.subheader(performance[i])
        st.header('Fitness')
        st.write(fitness)
        Schedule = create_schedule_visualization(results)
        st.pyplot(Schedule)

result_table = pd.read_excel(project_name + '.xlsx',sheet_name='ref_point', header=0,
                                     index_col=None)

if st.button('verify'):
    # show the schedule of the best solution
    st.write(len(result_table['# Solution']))
    for i in range(len(result_table['# Solution'])):
        best_solution = result_table.loc[i].at["best function"]
        st.write(best_solution)
        reference_point = [reference_points_new.iloc[0,i], reference_points_new.iloc[1, i], reference_points_new.iloc[2, i]]
        fitness, results, performance = verification(performance_measure=1, ref_point=True,
                                                     reference_point_input=reference_point,
                                                     best_func=best_solution, lambda_performance=weightings_ASF, nb_simulations=nb_simulations)

        wb = op.load_workbook(project_name + '.xlsx')
        ws_ref_point = wb["ref_point"]
        max_row = ws_ref_point.max_row
        ws_ref_point['D' + str(max_row + 1)] = performance[0]
        ws_ref_point['E' + str(max_row + 1)] = performance[1]
        ws_ref_point['F' + str(max_row + 1)] = performance[2]
        wb.save(project_name + '.xlsx')
        wb.close()

        st.header('Performance measures of the best solution')
        for i in range(len(performance_name)):
            st.subheader(performance_name[i])
            st.subheader(performance[i])
        st.header('Fitness')
        st.write(fitness)
        Schedule = create_schedule_visualization(results)
        st.pyplot(Schedule)

# check efficiency
makespan = result_table['Makespan']
tardiness = result_table['Total tardiness']
waiting = result_table['Total waiting time']
result_array = pd.concat([makespan, tardiness, waiting], axis=1, join="inner")
result_array = result_array.to_numpy()
if st.button('check dominance'):
    efficient_points = keep_efficient(result_array)
    st.write(efficient_points)
    efficient_points = is_pareto_efficient_dumb(result_array)
    st.write(efficient_points)

# visualize the result of the initialization phase
st.subheader('Relative performance of obtained solutions')

# create bar chart
number_solution = result_table[result_table['efficient']==True]['# Solution']
makespan = result_table[result_table['efficient']==True]['Makespan relativ']
tardiness = result_table[result_table['efficient']==True]['Total tardiness relativ']
waiting = result_table[result_table['efficient']==True]['Total waiting time relativ']
fig = create_relative_performance(makespan=makespan, tardiness=tardiness, waiting=waiting, x_labels=number_solution)
st.pyplot(fig)

st.header('Step 2: Exploration Phase')
st.write("""In the second phase of this procedure, you can further explore the objective space. 
Therefore, you can choose from following options:
* Add aspiration level to avoid undesirable regions of the objective space
* Intensification for exploring promising areas
""")

solution = st.selectbox('Please select the number of the solution to be evaluated:', (number_solution))
evaluation_options=['Add aspiration level to avoid undesirable regions of the objective space', 'Intensification for exploring promising areas']
option = st.selectbox('Please select the evaluation option:', evaluation_options)


if option == 'Add aspiration level to avoid undesirable regions of the objective space':

    constraint = st.selectbox('Please select the objective value you want to add a constraint:', performance_name)
    if (constraint == performance_name[0]):
        objective_constraint = 0
    if (constraint == performance_name[1]):
        objective_constraint = 1
    if (constraint == performance_name[2]):
        objective_constraint = 2

    value_reached = result_table[result_table['# Solution'] == solution][constraint]
    st.write('Performance of the selected solution: ', value_reached.iloc[0])

    selected_range = st.slider('Please select the allowed range of values (absolut)',
                               float(objective_ideal[objective_constraint]),
                               float(objective_nadir[objective_constraint]), (
                               float(objective_ideal[objective_constraint]),
                               float(objective_nadir[objective_constraint])))

    min_value = selected_range[0]
    max_value = selected_range[1]

    if st.button('Start Exploration'):

        reference_point = [reference_points_new.iloc[0,solution-1], reference_points_new.iloc[1, solution-1], reference_points_new.iloc[2, solution-1]]
        best_solution, best_fitness, nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues, stdFitnessValues = \
            geneticprogamming_simulation(performance_measure=1, ref_point=True, opt_type="best",
                                         project_name=project_name,
                                         reference_point_input=reference_point,
                                         nb_generations=nb_generations, population_size=population_size,
                                         crossover_probability=crossover_probability,
                                         mutation_probability=mutation_probability,
                                         max_depth_crossover=max_depth_crossover,
                                         max_depth_mutation=max_depth_mutation,
                                         nb_simulations=nb_simulations, lambda_performance=weightings_ASF, constraint_index=objective_constraint, constraint_value=max_value)

        # verify the solution and evaluate the performance measures
        fitness, results, performance = verification(performance_measure=1, ref_point=True,
                                                     reference_point_input=reference_point,
                                                     best_func=best_solution, lambda_performance=weightings_ASF,
                                                     nb_simulations=nb_simulations)
        # save the solution
        save_values(project_name=project_name, ref_point=True, performance_name=performance_name, performance_measure='1',
                    opt_type='best',
                    best_fitness=best_fitness, reference_point_input=reference_point, best_solution=best_solution,
                    performance=performance)

        # visualize GP statistics in form of a errorchart
        GP_statistics = create_statistics_visualization(nb_generation, avgFitnessValues, minFitnessValues,
                                                        maxFitnessValues)
        st.pyplot(GP_statistics)

        # create tree visualization of the best solution found
        image_tree = create_tree_visualization(best_solution)
        st.header('Primitive tree of the best solution')
        st.image(image_tree, use_column_width=True)

        # show the best solution found as a string representation
        st.header('Best dispatching rule found')
        st.write(str(best_solution))

        # show the fitness of the best solution
        st.header('Fitness of best solution')
        st.write(best_fitness)

        st.header('Performance measures of the best solution')
        for i in range(len(performance_name)):
            st.subheader(performance_name[i])
            st.subheader(performance[i])
        st.header('Fitness')
        st.write(fitness)
        Schedule = create_schedule_visualization(results)
        st.pyplot(Schedule)

        st.subheader('Comparison of the newly found solution to the original solution')

        # find the number of the last added solution
        result_table = pd.read_excel(project_name + '.xlsx', sheet_name='ref_point', header=0,
                                     index_col=None)
        solution2 = result_table.iloc[-1 , 0]

        # show radar chart
        fig = create_radar_chart(solution1=solution, solution2=solution2, objective_ideal=objective_ideal, objective_nadir=objective_nadir, project_name=project_name)
        st.pyplot(fig)

        st.write('If you want to discard one of the solutions, please select the solution and confirm')
        st.selectbox('Please select the solution you want to discard:', [solution, 'new solution'])
        st.button('Discard solution')

if option == 'Intensification for exploring promising areas':

    # select the option of the search
    st.selectbox('Please select the option for your search', ['Fast search', 'Extensive search'])

    # load reference point
    reference_point = [reference_points_new.iloc[0, solution - 1], reference_points_new.iloc[1, solution - 1],
                       reference_points_new.iloc[2, solution - 1]]
    # load solution performance
    makespan_solution = result_table[result_table['# Solution'] == solution]['Makespan']
    tardiness_solution = result_table[result_table['# Solution'] == solution]['Total tardiness']
    waiting_solution = result_table[result_table['# Solution'] == solution]['Total waiting time']
    solution_performance = [makespan_solution.iloc[0], tardiness_solution.iloc[0], waiting_solution.iloc[0]]
    # place the new reference point
    factor = st.select_slider('Please select the focus of the intensification:',
                              ['original reference point', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               'selected solution'])
    if factor == 'original reference point':
        factor = 0
    elif factor == 'selected solution':
        factor = 1
    fig, reference_point_intensification = create_visualization_zoom(reference_point=reference_point, solution=solution_performance, factor=factor)
    st.pyplot(fig)



    weightings_intensification = [[4 / 3 * weightings_ASF[0], 5 / 6 * weightings_ASF[1], 5 / 6 * weightings_ASF[2]],
                                  [5 / 6 * weightings_ASF[0], 4 / 3 * weightings_ASF[1], 5 / 6 * weightings_ASF[2]],
                                  [5 / 6 * weightings_ASF[0], 5 / 6 * weightings_ASF[1], 4 / 3 * weightings_ASF[2]],
                                  [7 / 6 * weightings_ASF[0], 7 / 6 * weightings_ASF[1], 2 / 3 * weightings_ASF[2]],
                                  [7 / 6 * weightings_ASF[0], 2 / 3 * weightings_ASF[1], 7 / 6 * weightings_ASF[2]],
                                  [2 / 3 * weightings_ASF[0], 7 / 6 * weightings_ASF[1], 7 / 6 * weightings_ASF[2]]]

    if st.button('Start Exploration'):

        # run the GP-HH based Schedule Generator for 3 different weightings
        for w in range(6):
            best_solution, best_fitness, nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues, stdFitnessValues = \
                geneticprogamming_simulation(performance_measure=1, ref_point=True, opt_type="best",
                                             project_name=project_name,
                                             reference_point_input=reference_point_intensification,
                                             nb_generations=nb_generations, population_size=population_size,
                                             crossover_probability=crossover_probability,
                                             mutation_probability=mutation_probability,
                                             max_depth_crossover=max_depth_crossover,
                                             max_depth_mutation=max_depth_mutation,
                                             nb_simulations=nb_simulations, lambda_performance=weightings_intensification[w],
                                             constraint_index=None, constraint_value=None)

            # verify the solution and evaluate the performance measures
            fitness, results, performance = verification(performance_measure=1, ref_point=True,
                                                         reference_point_input=reference_point,
                                                         best_func=best_solution, lambda_performance=weightings_ASF,
                                                         nb_simulations=nb_simulations)
            # save the solution
            save_values(project_name=project_name, ref_point=True, performance_name=performance_name,
                        performance_measure='1',
                        opt_type='best',
                        best_fitness=best_fitness, reference_point_input=reference_point, best_solution=best_solution,
                        performance=performance)

            # visualize GP statistics in form of a errorchart
            GP_statistics = create_statistics_visualization(nb_generation, avgFitnessValues, minFitnessValues,
                                                            maxFitnessValues)
            st.pyplot(GP_statistics)

            # create tree visualization of the best solution found
            image_tree = create_tree_visualization(best_solution)
            st.header('Primitive tree of the best solution')
            st.image(image_tree, use_column_width=True)

            # show the best solution found as a string representation
            st.header('Best dispatching rule found')
            st.write(str(best_solution))

            # show the fitness of the best solution
            st.header('Fitness of best solution')
            st.write(best_fitness)

            st.header('Performance measures of the best solution')
            for i in range(len(performance_name)):
                st.subheader(performance_name[i])
                st.subheader(performance[i])
            st.header('Fitness')
            st.write(fitness)
            Schedule = create_schedule_visualization(results)
            st.pyplot(Schedule)




st.header('Step 3: Decision Phase')
st.write("""This step will help to find the most suitable solution for you. 
Through a pairwise comparison of all the explored solutions the "best" compromise can be identified. 
In a step by step procedure you will be asked to decide which of both solutions shown you prefer. 
The solution that looses the comparison will be eliminated from the set of acceptable solutions. 
This procedure will be repeated until a single solution is left. If this solutions satisfies your expectations, the
whole procedure successfully guided you to your best solution. Otherwise you can start from the beginning or explore more new solutions.
""")

# load the set of nondominated solutions
set_of_nondominated_solutions = result_table[result_table['efficient']==True]
set_of_nondominated_solutions = set_of_nondominated_solutions[set_of_nondominated_solutions['elimination']==0]


# check efficiency
makespan = set_of_nondominated_solutions['Makespan']
tardiness = set_of_nondominated_solutions['Total tardiness']
waiting = set_of_nondominated_solutions['Total waiting time']
result_array = pd.concat([makespan, tardiness, waiting], axis=1, join="inner")
result_array = result_array.to_numpy()
if st.button('check dominance 2'):
    efficient_points = keep_efficient(result_array)
    st.write(efficient_points)
    efficient_points = is_pareto_efficient_dumb(result_array)
    st.write(efficient_points)

# create table

results_table = pd.read_excel(project_name + '.xlsx',sheet_name='results', header=0,
                                     index_col=None)

st.write(results_table)

# create bar chart
number_solution = set_of_nondominated_solutions['# Solution']
makespan = set_of_nondominated_solutions['Makespan relativ']
tardiness = set_of_nondominated_solutions['Total tardiness relativ']
waiting = set_of_nondominated_solutions['Total waiting time relativ']
fig = create_relative_performance(makespan=makespan, tardiness=tardiness, waiting=waiting, x_labels=number_solution)
st.pyplot(fig)

# randomly select two solutions
solution_list = number_solution.values.tolist()
solutions_to_compare = random.sample(solution_list,2)

solution1 = solutions_to_compare[0]
solution2 = solutions_to_compare[1]

st.subheader(f'Compare solution # {solution1} to solution # {solution2}')

fig = create_radar_chart(project_name=project_name, solution1=solution1, solution2=solution2, objective_ideal=objective_ideal, objective_nadir=objective_nadir)
st.pyplot(fig)

st.write('Which solution do you prefer? If you do not have a clear preference you can also choose to skip')
if st.button(f'Solution #{solution1}'):
    eliminate_solution(project_name=project_name, solution=solution1)

if st.button(f'Solution #{solution2}'):
    eliminate_solution(project_name=project_name, solution=solution2)

st.button(f'Skip to the next pair')


st.header('Test and Comparison of the final solution to other benchmark dispatching rules')

final_results = pd.read_excel(project_name + '.xlsx',sheet_name='final_results', header=0,
                                     index_col=None)

final_results_training = final_results[final_results['comment']=='train set']
final_results_testing = final_results[final_results['comment']=='test set']


# create bar chart for Training set
st.subheader('Training set')
number_solution = final_results_training['# Solution']
makespan = final_results_training['Makespan relativ']
tardiness = final_results_training['Total tardiness relativ']
waiting = final_results_training['Total waiting time relativ']
fig = create_relative_performance_final(makespan=makespan, tardiness=tardiness, waiting=waiting, x_labels=number_solution, header='Relative performance on the training set')
st.pyplot(fig)

# create bar chart for Testing set
st.subheader('Testing set')
number_solution = final_results_testing['# Solution']
makespan = final_results_testing['Makespan relativ']
tardiness = final_results_testing['Total tardiness relativ']
waiting = final_results_testing['Total waiting time relativ']
fig = create_relative_performance_final(makespan=makespan, tardiness=tardiness, waiting=waiting, x_labels=number_solution,header='Relative performance on the test set')
st.pyplot(fig)
'''
# best solution in form of a parse tree
best_solution = final_results_training[final_results_training['# Solution']=="Solution #32"]['best function']
best_solution = best_solution.iloc[0]
st.subheader('Final preference-based dispatching rule')
st.write(best_solution)

# verify the solution and evaluate the performance measures
reference_point = [0,0,0]
fitness, results, performance, func = verification_final(performance_measure=1, ref_point=True,
                                             reference_point_input=reference_point,
                                             best_func=best_solution, lambda_performance=weightings_ASF,
                                             nb_simulations=nb_simulations)

fig = create_tree_visualization(best_solution=func)
st.pyplot(fig)
'''
# visualize the final parse tree
# load parse tree data
labels = pd.read_excel('Parse_Tree_Solution_37.xlsx',sheet_name='labels', header=None,
                                     index_col=0, converters={0:str,1:str})
edges = pd.read_excel('Parse_Tree_Solution_37.xlsx',sheet_name='edges', header=None,
                                     index_col=0)
nodes = pd.read_excel('Parse_Tree_Solution_37.xlsx',sheet_name='nodes', header=None,
                                     index_col=0)

labels = labels.to_dict()
labels = labels[1]
edges = edges.values.tolist()
nodes = nodes.T
nodes = nodes.values.tolist()
nodes = nodes[0]
st.write(labels)
st.write(edges)
st.write(nodes)

image_tree = create_tree_visualization_final_solution(nodes, edges, labels)
st.image(image_tree, use_column_width=True)

'''
st.header('Select the reference point')
reference_point_selection = [0,0,0]
reference_point_selection[0] = st.slider('Makespan', min_value=float(objective_ideal[0]), max_value=float(objective_nadir[0]))
reference_point_selection[1] = st.slider('Number of tardy jobs', min_value=float(objective_ideal[1]), max_value=float(objective_nadir[1]))
reference_point_selection[2] = st.slider('Total tardiness', min_value=float(objective_ideal[2]), max_value=float(objective_nadir[2]))


st.header('Evolve dispatching rule')
if st.button("run"):
    best_solution, best_fitness, nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues, stdFitnessValues = \
        geneticprogamming_simulation(performance_measure=1, ref_point=True, opt_type="best", project_name=project_name, reference_point_input=reference_point_selection,
                      nb_generations=nb_generations, population_size=population_size,
                      crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                      max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                      nb_simulations=nb_simulations, lambda_performance=weightings_ASF)


    # visualize GP statistics in form of a errorchart
    GP_statistics = create_statistics_visualization(nb_generation, avgFitnessValues, minFitnessValues, maxFitnessValues)
    st.pyplot(GP_statistics)

    # create tree visualization of the best solution found
    image_tree = create_tree_visualization(best_solution)
    st.header('Primitive tree of the best solution')
    st.image(image_tree,  use_column_width=True)

    # show the best solution found as a string representation
    st.header('Best dispatching rule found')
    st.write(str(best_solution))

    # show the fitness of the best solution
    st.header('Fitness of best solution')
    st.write(best_fitness)

    # show the schedule of the best solution
    fitness, results, performance = verification(performance_measure=1, ref_point=True, reference_point_input=reference_point_selection, best_func=best_solution, lambda_performance=weightings_ASF)
    st.header('Performance measures of the best solution')
    for i in range(len(performance_name)):
        st.subheader(performance_name[i])
        st.subheader(performance[i])
    st.header('Fitness')
    st.write(fitness)
    Schedule = create_schedule_visualization(results)
    st.pyplot(Schedule)
'''


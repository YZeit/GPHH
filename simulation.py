import numpy as np
import time
import pandas as pd


def simulation(number_machines, number_jobs, func, random_seed):

    schedule = []
    results = []
    global_clock = 0
    np.random.seed(random_seed)

    #number_machines = 10
    number_operations = number_machines
    #number_jobs = 10
    due_date_tightness = 1.3
    TRNO = number_operations*number_jobs

    class Job():
        def __init__(self):
            self.start = 0
            self.end = 0
            self.clock = 0
            self.operations = []
            self.number_operations = 0
            self.RPT = 0
            self.RNO = 0
            self.DD = 0
            self.operation_to_release = int(0)
            self.release_status = 'no'
            self.t_event = 0
            self.number = 0

        class Operation():
            def __init__(self):
                self.number = 0
                self.start = 0
                self.end = 0
                self.clock = 0
                self.PT = 0
                self.machine = int(999999)

    class Machine():
        def __init__(self):
            #self.queuing_dict = {"machine":[], "job":[], "prec":[], "PT":[], "RPT":[], "DD":[], "RNO":[], "priority":[]}
            self.queue = {'Job':[], 'Operation':[], 'Priority':[]}
            self.job_to_release = []
            self.num_in_system = 0
            self.clock = 0.0
            self.t_depart = float('inf')
            self.t_event = 0
            self.status = 'Idle'
            self.current_job_finish = 0
            self.TRNO = 0
            self.SPT = 0

        def execute(self):
            # update priority
            self.update_priority()
            # select the waiting operation with the lowest priority value
            min_priority = min(self.queue["Priority"])
            index_job = self.queue["Priority"].index(min_priority)
            #print(self.queue['Job'][index_job].number)
            # update operation and job data
            self.queue['Operation'][index_job].start = self.clock
            self.queue['Operation'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].t_event = self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].clock += self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].RPT -= self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].RNO -= 1
            if self.queue['Operation'][index_job].number == 0:
                self.queue['Job'][index_job].start = self.clock
            if self.queue['Operation'][index_job].number == (number_operations-1):
                self.queue['Job'][index_job].end = self.clock + self.queue["Operation"][index_job].PT

            self.t_event = self.queue["Operation"][index_job].PT
            self.clock += self.t_event

            # save data in schedule and result dict
            schedule.append([self.queue["Job"][index_job].number, self.queue["Operation"][index_job].machine])
            results.append({'Job': f'J{self.queue["Job"][index_job].number}',
                            'Machine': f'M{self.queue["Operation"][index_job].machine}',
                            'Start': self.clock - self.t_event,
                            'Duration': self.t_event,
                            'Finish': self.clock})
            self.current_job_finish = self.clock


            # set job status to 'release'
            self.queue['Job'][index_job].operation_to_release += 1
            self.queue['Job'][index_job].release_status = 'yes'
            self.queue['Job'][index_job].clock = self.clock


            # remove operation from queue
            del self.queue["Job"][index_job]
            del self.queue["Operation"][index_job]
            del self.queue["Priority"][index_job]

            # set status to 'running'
            self.status = 'Running'



        def update_priority(self):

            '''validation_dict = {"Job": [], 'Operation': [], 'Machine': [], 'Time': [], 'PT': [], 'RPT': [], 'RNO': [],
                               'DD': [],
                               'SPTQ': [], 'APTQ': [], 'MAXPTQ': [], 'MINPTQ': [], 'MAXDDQ': [], 'NJQ': [], 'SPT': [],
                               'TRNO': [],
                               'CT': [], 'dispatching rule': [], 'Priotity': []}'''

            PT_list = []
            DD_list = []
            for i in range(len(self.queue['Job'])):
                PT_list.append(self.queue['Operation'][i].PT)
                DD_list.append(self.queue['Job'][i].DD)
            SPTQ = np.sum(PT_list)
            APTQ = np.mean(PT_list)
            MAXPTQ = np.max(PT_list)
            MINPTQ = np.min(PT_list)
            MAXDDQ = np.max(DD_list)
            NJQ = len(self.queue['Job'])
            TRNO = self.TRNO
            SPT = self.SPT
            for i in range(len(self.queue['Job'])):
                DD = self.queue['Job'][i].DD
                PT = self.queue['Operation'][i].PT
                RPT = self.queue['Job'][i].RPT
                RNO = self.queue['Job'][i].RNO
                CT = self.clock
                priority = func(PT, RPT, RNO, DD, SPTQ, APTQ, MAXPTQ, MINPTQ, MAXDDQ, NJQ, SPT, TRNO, CT)
                #priority = PT
                self.queue["Priority"][i] = priority

                '''validation_dict['Job'].append(self.queue['Job'][i].number)
                validation_dict['Operation'].append(self.queue['Operation'][i].number)
                validation_dict['Machine'].append(self.queue['Operation'][i].machine)
                validation_dict['Time'].append(self.clock)
                validation_dict['PT'].append(PT)
                validation_dict['RPT'].append(RPT)
                validation_dict['RNO'].append(RNO)
                validation_dict['DD'].append(DD)
                validation_dict['SPTQ'].append(SPTQ)
                validation_dict['APTQ'].append(APTQ)
                validation_dict['MAXPTQ'].append(MAXPTQ)
                validation_dict['MINPTQ'].append(MINPTQ)
                validation_dict['MAXDDQ'].append(MAXDDQ)
                validation_dict['NJQ'].append(NJQ)
                validation_dict['SPT'].append(SPT)
                validation_dict['TRNO'].append(TRNO)
                validation_dict['CT'].append(CT)
                validation_dict['dispatching rule'].append(str(func))
                validation_dict['Priotity'].append(priority)
            validation_pd = pd.DataFrame(validation_dict)
            print(validation_pd.to_string())'''

    def load_random_test_probem(jobs, due_date_tightness):
        # random instance generator according to Taillard
        average_processing_time = 0
        totaltotal_processing_time = 0
        for j in jobs:
            # allowed values for machines (after each iteration a machine will be discarded from the set
            # in order to ensure that each machines has been assigned to a job
            allowed_values = list(range(0, number_machines))
            total_processing_time = 0
            for o in j.operations:
                o.PT = np.random.uniform(1, 99)
                # print(o.PT)
                o.machine = np.random.choice(allowed_values)
                total_processing_time += o.PT
                o.number = j.operations.index(o)
                allowed_values.remove(o.machine)

            j.number = jobs.index(j)

            # Due Date according to flow shop problem (evolutionary algorithm book)
            #totaltotal_processing_time += total_processing_time
            #average_processing_time = totaltotal_processing_time/((jobs.index(j)+1)*number_machines)
            #j.DD = np.random.uniform(average_processing_time*number_machines, average_processing_time*(number_jobs+number_machines-1))
            #print(j.number)
            #print(j.DD)

            # Due Date according to Baker 1984
            j.DD = due_date_tightness * total_processing_time

            j.RPT = total_processing_time
            j.RNO = len(j.operations)
            j.release_status = 'yes'


    def load_benchmark_problem():
        # TA20

        # TA20 benchmark problem
        number_machines, number_jobs = 15, 20
        data = """
                 7 84  0 58 12 71  4 26  1 98  9 36  2 12 11 30 10 87 14 95  5 45  6 28 13 73  3 73  8 45 
                 4 29  8 22  7 47  3 75  9 94 13 15 12  4  0 82 11 14 10 35  1 79  6 34  5 57 14 23  2 56 
                 1 73  4 36  7 48 13 26  3 49  8 60 10 15  5 66 12 90 14 39  9  8  6 74  2 63  0 94 11 91 
                 5  1 11 35  9 23 12 93  7 75  1 50  6 40 13 60  8 41  2  7  0 57 14 72  3 40  4 75 10  7 
                 4 13 11 15 12 17  1 14  0 67  9 94  6 18 13 52  2 53 14 16  5 33 10 61  3 47  8 65  7 39 
                 2 54  6 80  3 87  8 36 14 54  0 72  4 17 10 44 11 37  1 88  7 77 13 84 12 17  5 82  9 90 
                 4  4 14 62  5 33 10 62  8 86  7 30  6 39  1 67  0 42 12 31  9 83 13 39 11 67  3 67  2 31 
                 7 29 10 29 11 69 14 26  3 55  2 46  4 53  5 65  1 97 12 24  9 69  6 22 13 17  0 39  8 13 
                14 12 11 73  0 36 13 70  3 12  2 80  1 99  8 70  5 51  7 14  4 71 12 28  6 35 10 58  9 35 
                 0 61  5 49 12 74  1 90 13 60 10 88  9  3  4 60  2 59  8 94 14 91 11 34  7 26  6  4  3 26 
                 4 89  3 90  8 95 12 32  9 18 11 73  2  9 14 19  5 97  7 58 13 36  6 62 10 13  1 16  0  1 
                 9 71  6 47  1 95  0  7 14 63  7 49 13 24 12 46  2 72 11 73  5 19  8 96 10 41  3 15  4 81 
                 4 45  3  9  0 97 14 62 13 77  9 78  7 70  2 19 11 86  8 15 10 23  1 46  6 32 12  6  5 70 
                12 74 10 46  3 98  6  1  4 53  5 59  0 86  7 98  2 76  8 12 13 91 11 98 14 98  9 11  1 27 
                14 73  7 70  5 14  8 32 11 19  0 57  2 17 13 96 12 56  4 73  6 32  1  7 10 79  9 10  3 91 
                 6 39 14 87 12 11  2 81  7  7  5 79  8 24 13  9 11 58  9 42  0 67  3 27  4 20  1 19 10 67 
                 9 76  5 89 14 64 10 14 12 11  1 14  4 99 13 85  0 81 11  3  3 46  2 47  7 40  6 81  8 27 
                 9 55 12 71  4  5 14 83 11 16  8  4  0 20  7 15  5 60  3  8  1 93 10 33  6 63 13 71  2 29 
                12 92  2 25  3  8 14 86  5 22  1 79  6 23 11 96 13 24  9 94  7 97 10 17  8 48  0 67  4 47 
                 3  5 12 77 10 74  5 59 14 13  0 57  9 62  8 37 13 54  6 69 11 80  1 35  7 88  2 47  4 98 
                     """
        TASKS = {}
        for job, line in enumerate(data.splitlines()[1:]):
            nums = line.split()
            prec = None
            for m, dur in zip(nums[::2], nums[1::2]):
                task = (job, int(m))
                TASKS[task] = {'dur': int(dur), 'prec': prec}
                prec = task

        TASKS = pd.DataFrame(TASKS)



        #total_processing_time = sum(TASKS[(j, m)]['dur'] for m in range(number_machines) for j in range(number_jobs))

        for j in jobs:

            # allowed values for machines (after each iteration a machine will be discarded from the set
            # in order to ensure that each machines has been assigned to a job
            j.number = jobs.index(j)
            total_processing_time_job = sum(
                TASKS[(j.number, m)]['dur'] for m in range(number_machines))
            #print('Job: ' + str(j.number))
            TASKS_Job = TASKS[j.number]
            TASKS_Job = TASKS_Job.T

            for o in j.operations:
                o.number = j.operations.index(o)
                #print('Operation: ' + str(o.number))
                TASKS_Operation = TASKS_Job.iloc[o.number]
                machine = TASKS_Operation.name
                duration = TASKS_Operation.dur
                o.PT = duration
                #print('Duration: ' + str(duration))
                #print('Machine: ' + str(machine))
                o.machine = machine
            j.DD = due_date_tightness * total_processing_time_job
            j.RPT = total_processing_time_job
            j.RNO = len(j.operations)
            j.release_status = 'yes'

    # measure performance of the simulation
    start = time.time()

    # generate machines
    machines = [Machine() for m in range(number_machines)]

    # generate jobs
    jobs = [Job() for j in range(number_jobs)]

    # define number of operations for each job (here equal to number of machines)
    # generate operation class
    for j in jobs:
        j.number_operations=number_machines
        j.operations = [j.Operation() for o in range(j.number_operations)]

    # load the problem instance
    load_random_test_probem(jobs, due_date_tightness)
    #load_benchmark_problem()

    # calculate the total processing time in the problem
    SPT = 0
    for j in jobs:
        for o in j.operations:
            SPT += o.PT
    for m in machines:
        m.SPT = SPT
        m.TRNO = TRNO

    # start simulation
    # first round
    # initial release of the first operations of each job
    for j in jobs:
        if j.release_status == 'yes':
                number_of_released_operation = j.operation_to_release
                machine_to_release = j.operations[number_of_released_operation].machine
                machines[machine_to_release].queue['Job'].append(j)
                machines[machine_to_release].queue['Operation'].append(j.operations[number_of_released_operation])
                machines[machine_to_release].queue['Priority'].append(0)
                j.release_status='no'

    # calculate number of waiting jobs
    number_of_waiting_jobs = 0
    for i in machines:
        number_of_waiting_jobs += len(i.queue["Job"])
        #print(number_of_waiting_jobs)
    # loop while there are jobs waiting in the system
    while TRNO > 0:
        # check if there are operations to be released on each job
        for j in jobs:
            if j.clock <= global_clock:
                if j.release_status == 'yes':
                    number_of_released_operation = j.operation_to_release
                    if number_of_released_operation <= len(j.operations)-1:
                        #print(number_of_released_operation)
                        machine_to_release = j.operations[number_of_released_operation].machine
                        machines[machine_to_release].queue['Job'].append(j)
                        machines[machine_to_release].queue['Operation'].append(j.operations[number_of_released_operation])
                        machines[machine_to_release].queue['Priority'].append(0)
                        j.release_status = 'no'

        # check if there are jobs waiting in the queue on each machine
        for i in machines:
            if i.clock <= global_clock:
                #print(i.queuing_dict)
                if len(i.queue["Job"]) != 0:
                    i.TRNO = TRNO
                    i.SPT = SPT
                    i.execute()
                    TRNO -= 1
                    SPT -= i.t_event

        # check for next event on both classes (jobs and machines)
        t_next_event_list = []
        for m in machines:
            if m.clock > global_clock:
                t_next_event_list.append(m.clock)
        for j in jobs:
            if j.clock > global_clock:
                t_next_event_list.append(j.clock)

        # next event and update of global clock
        if t_next_event_list != []:
            t_next_event = min(t_next_event_list)
        else:
            t_next_event=0
        global_clock=t_next_event
        # calculate number of waiting jobs
        number_of_waiting_jobs = 0
        for i in machines:
            number_of_waiting_jobs += len(i.queue["Job"])

        # set the machine times less than the global time to the global time
        for i in machines:
            if i.clock <= global_clock:
                i.clock = global_clock
        for j in jobs:
            if j.clock <= global_clock:
                j.clock = global_clock

    end = time.time()
    #print('time needed \t\t', end-start)

    # create visualization of the schedule
    #Schedule = create_schedule_visualization(results)
    #plt.show()

    # calculate performance measures
    # makespan
    schedule = pd.DataFrame(results)
    #print(schedule[schedule['Job']=='J9'])
    makespan = schedule['Finish'].max()
    # tardiness
    tardiness = 0
    earliness = 0
    waiting_time = 0
    for j in jobs:
        tardiness += max((j.end - j.DD), 0)
        earliness += max((j.DD - j.end, 0))
        waiting_time += j.end-j.start


    # print performance measures
    #print('Makespan: ' + str(makespan))
    #print('Tardiness: ' + str(tardiness))
    #print('Earliness: ' + str(earliness))

    return results, makespan, tardiness, waiting_time
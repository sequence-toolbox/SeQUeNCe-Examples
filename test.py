"""
This replicates the experiment Fig.6 https://arxiv.org/pdf/2504.01290

Author: R. J. Hayek, Argonne National Laboratory
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sequence.app.request_app import RequestApp
from sequence.components.memory import Memory
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM, SINGLE_HERALDED
from sequence.entanglement_management.generation import EntanglementGenerationA, SingleHeraldedA, \
    EntanglementGenerationB
from sequence.entanglement_management.purification import BBPSSWMessage, BBPSSW_BDS, BBPSSWMsgType, BBPSSWProtocol
from sequence.kernel.quantum_manager import QuantumManager as qm
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.utils import log


@BBPSSWProtocol.register("TrackingBBPSSW")
class TrackingBBPSSW(BBPSSW_BDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def received_message(self, src: str, msg: BBPSSWMessage) -> None:
        """Method to receive messages. Adds tracking

        args:
            src (str): name of node that sent the message.
            msg (BBPSSW message): message received.

        Side Effects:
            Will call `update_resource_manager` method.
        """

        # check the status of entanglement
        if self.meas_memo.entangled_memory['node_id'] is None or self.kept_memo.entangled_memory['node_id'] is None:
            log.logger.info(f'No entanglement for {self.meas_memo} or {self.kept_memo}.')
            # when the AC Protocol expires, the purification protocol on the primary node will get removed, but the purification protocol on the non-primary node is still there
            self.owner.protocols.remove(self)
            return

        if msg.msg_type == BBPSSWMsgType.PURIFICATION_RES:

            purification_success = (self.meas_res == msg.meas_res)
            log.logger.info(self.owner.name + f" received result message, succeeded={purification_success}")
            assert src == self.remote_node_name

            self.update_resource_manager(self.meas_memo, "RAW")

            if purification_success:
                log.logger.info(f'Purification success, measurement results: {self.meas_res}, {msg.meas_res}')
                self.owner.ep_success +=1
                remote_kept_memory_name = self.remote_memories[0]
                remote_kept_memory: Memory = self.owner.timeline.get_entity_by_name(remote_kept_memory_name)
                remote_kept_memory.bds_decohere()
                self.kept_memo.bds_decohere()
                self.kept_memo.fidelity = self.kept_memo.get_bds_fidelity()
                self.owner.new_fid = self.kept_memo.fidelity
                self.update_resource_manager(self.kept_memo, state="ENTANGLED")

                if self.owner.ep_success == 500:
                    self.owner.time_to_ep = self.owner.timeline.now() - 1e12

            else:
                log.logger.info(f'Purification failed because measure results: {self.meas_res}, {msg.meas_res}')
                self.update_resource_manager(self.kept_memo, state="RAW")

        else:
            raise Exception(f'{msg.msg_type} unknown')

    def start(self) -> None:
        self.owner.ep_count += 1
        super().start()

@EntanglementGenerationA.register("TrackingEntanglement")
class TrackingEntanglement(SingleHeraldedA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _entanglement_succeed(self):
        self.owner.total_attempts += 1
        self.owner.successful_attempts += 1
        if self.owner.successful_attempts == 1000:
            self.owner.time_to_thousand = self.owner.timeline.now() - 1e12
        super()._entanglement_succeed()

    def _entanglement_fail(self):
        self.owner.total_attempts += 1
        super()._entanglement_fail()


def bananas():
    _original_init = QuantumRouter.__init__
    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.successful_attempts = 0
        self.total_attempts = 0
        self.time_to_thousand = 0
        self.ep_count = 0
        self.ep_success = 0
        self.time_to_ep = 0
        self.new_fid = 0
    QuantumRouter.__init__ = _patched_init


def _run_trial(CONFIG_FILE, PREP_TIME, COLLECT_TIME, QC_FREQ, APP_NODE_NAME,
               OTHER_NODE_NAME, NUM_MEMORIES, fidelity, trial) -> tuple:
    EntanglementGenerationA.set_global_type('TrackingEntanglement')
    EntanglementGenerationB.set_global_type(SINGLE_HERALDED) # Must do this if changing A type
    BBPSSWProtocol.set_formalism('TrackingBBPSSW')


    bananas()
    # establish network
    net_topo = RouterNetTopo(CONFIG_FILE)
    qm.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)

    # timeline setup
    tl = net_topo.get_timeline()
    tl.stop_time = PREP_TIME + COLLECT_TIME

    # network configuration
    routers = net_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    bsm_nodes = net_topo.get_nodes_by_type(RouterNetTopo.BSM_NODE)

    base_seed = int(time.time_ns()) % (2 ** 20) + os.getpid()
    # Random seed for performing the simulations
    for j, node in enumerate(routers + bsm_nodes):
        node.set_seed(base_seed + trial * 1000 + j)

    # set quantum channel parameters
    for qc in net_topo.get_qchannels():
        qc.frequency = QC_FREQ

    # establish "left" node as the start node.
    start_node = None
    for node in routers:
        if node.name == APP_NODE_NAME:
            start_node = node
            break
    # Checking to see if the start node was established or not
    if not start_node:
        raise ValueError(f"Invalid app node name {APP_NODE_NAME}")

    # Setting the "right" node as the 'end' node
    end_node = None
    for node in routers:
        if node.name == OTHER_NODE_NAME:
            end_node = node
            break
    # Checking to see if the end node was established or not
    if not start_node:
        raise ValueError(f"Invalid other node name {OTHER_NODE_NAME}")

    # Establishing the apps on the start and end nodes.
    app_start = RequestApp(start_node)
    RequestApp(end_node)  # This call is NECESSARY, though unassigned

    # initialize and start app
    tl.init()
    app_start.start(OTHER_NODE_NAME, PREP_TIME, PREP_TIME + COLLECT_TIME, NUM_MEMORIES, fidelity+0.01)
    tl.run()

    # Used for debugging
    attempt = app_start.node.total_attempts
    success = app_start.node.successful_attempts
    tteg = app_start.node.time_to_thousand
    ttep = app_start.node.time_to_ep

    success_rate = app_start.node.ep_success / app_start.node.ep_count

    final_fidelity = app_start.node.new_fid
    throughput = app_start.get_throughput()

    return tteg, ttep, success_rate, final_fidelity, throughput


def modify_config(config_file, decoherence, fidelity):
    """
    Modify the configuration file to set decoherence and initial fidelity.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)

    config['templates']['perfect_router']['MemoryArray']['coherence_time'] = decoherence
    config['templates']['perfect_router']['MemoryArray']['fidelity'] = fidelity

    with open(config_file, 'w') as file:
        json.dump(config, file, indent=4)


def main():
    bananas()  # Monkey patch to track attempts and successes
    print("Starting the experiment.")
    CONFIG_FILE = "ep_config.json"

    NO_TRIALS = 10

    # simulation params
    NUM_MEMORIES = 2
    PREP_TIME = int(1e12)  # 1 second
    COLLECT_TIME = int(100e12) / NUM_MEMORIES  # 10 seconds

    # qc params
    QC_FREQ = 1e11

    # application params
    APP_NODE_NAME = "left"
    OTHER_NODE_NAME = "right"

    initial_fidelities = np.round(np.arange(0.6, 0.9, 0.05), 3)
    decoherences = [18e-3, 55e-3, -1]
    data_dict = {
        'Decoherence': [],
        'Initial Fidelity': [],
        'Average EG Time': [],
        'Average EP Time': [],
        'Average Success Rate': [],
        'Average Final Fidelity': [],
        'Average Throughput': []
    }

    for decoherence in decoherences:
        for i, fidelity in enumerate(initial_fidelities):
            print(f"Running {NO_TRIALS} trials for decoherence={decoherence}, fidelity={fidelity}")
            modify_config(CONFIG_FILE, decoherence=decoherence, fidelity=fidelity)

            data_dict['Decoherence'].append(decoherence)
            data_dict['Initial Fidelity'].append(fidelity)

            ttegs = np.zeros(NO_TRIALS)
            tteps = np.zeros(NO_TRIALS)
            rates = np.zeros(NO_TRIALS)
            final_fidelities = np.zeros(NO_TRIALS)
            throughputs = np.zeros(NO_TRIALS)


            params = [(CONFIG_FILE, PREP_TIME, COLLECT_TIME, QC_FREQ, APP_NODE_NAME,
                OTHER_NODE_NAME, NUM_MEMORIES, fidelity, trial) for trial in range(NO_TRIALS)]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(_run_trial, params)

            for j, (tteg, ttep, success_rate, final_fidelity, throughput) in enumerate(results):
                ttegs[j] = tteg
                tteps[j] = ttep
                rates[j] = success_rate
                final_fidelities[j] = final_fidelity
                throughputs[j] = throughput

            data_dict['Average EG Time'].append(np.mean(ttegs) * 1e-12)
            data_dict['Average EP Time'].append(np.mean(tteps) * 1e-12)
            data_dict['Average Success Rate'].append(np.mean(rates))
            data_dict['Average Final Fidelity'].append(np.mean(final_fidelities))
            data_dict['Average Throughput'].append(np.mean(throughputs))

            print(f"Finished trials for decoherence={decoherence}, fidelity={fidelity}")

    print('Experiment completed. Saving results to CSV.')
    df = pd.DataFrame(data_dict)
    df.to_csv(f'experiment_results_{NO_TRIALS}.csv', index=False)

if __name__ == "__main__":
    main()
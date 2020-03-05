import dyPolyChord
import mpi4py


def nested_sampler(my_callable, dynamic_goal, settings_dict, ninit, nlive_const, seed_inc):

	comm = MPI.COMM_WORLD

	dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict, ninit=ninit, nlive_const=nlive_const, seed_increment=5, comm=comm)
	print('Hello world')

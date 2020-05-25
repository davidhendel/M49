import os
import sys
import time
import numpy as np
import scipy
import amuse.io
from amuse.lab import *
from amuse.datamodel import Particles
from amuse.community.bhtree.interface import BHTree
from amuse.ext.bridge import bridge
from amuse.support.literature import LiteratureReferencesMixIn
from amuse.datamodel import Particles
from amuse.units import quantities
from amuse.community.halogen.interface import Halogen
from amuse.community.gadget2.interface import Gadget2

#set environment variables for your directory structure in .bash_profile and .bash_rc, e.g.
#    export _M49_DATA_DIR=/path/to/M49data/
#if you are having problems with "not enough processors" spawn errors, 
#try "export OMPI_MCA_rmaps_base_oversubscribe=1"

_DATADIR  =  os.environ['_M49_DATA_DIR']
_LOCALDIR =  os.environ['_M49_LOCAL_DIR']



def initialize_nfw_model(N, mass = 1.77e12, scale_radius=24.6, cutoff_mod = 15., outfile=None):
    '''mass in Msun
    radius in Kpc
    Halogen: M Zemp et al. 2008 
    default_options = dict(redirection = "none")
    NFW = (1,3,1)
    need cutoff for beta <=3, gamma must be <3'''
    
    mass = mass | units.MSun
    scale_radius= scale_radius | units.kpc
    converter=nbody_system.nbody_to_si(mass, scale_radius)
    
    instance = Halogen(converter)
    instance.initialize_code()
    instance.parameters.alpha = 1.0
    instance.parameters.beta  = 3.0
    instance.parameters.gamma = 1.0
    instance.parameters.scale_radius = scale_radius
    instance.parameters.cutoff_radius = cutoff_mod * instance.parameters.scale_radius
    instance.parameters.number_of_particles = N
    instance.parameters.random_seed = 1
    instance.commit_parameters()
    instance.generate_particles()
    stars = instance.particles.copy()
        
    instance.cleanup_code()
    instance.stop_reusable_channels()
    instance.stop()
    if outfile is not None: amuse.io.write_set_to_file(stars, outfile, format='csv')
        
    return stars, converter

if __name__ == "__main__":

    tend = 2.94 | units.Gyr
    dt = 1. | units.Myr
    # max possible evolve time for Gadget, 2**k *dt should be > tend
    time_max = 2**12*dt
    assert(time_max > tend)

    particles_gal, converter_gal = initialize_nfw_model(5e6, mass = 10**15.4, scale_radius=10**2.8)
    particles_sat, converter_sat = initialize_nfw_model(1e6, mass = 1e12, scale_radius=10., cutoff_radius = 5.)

    particles_sat.x += 63.25 | units.kpc
    particles_sat.y += 78.14 | units.kpc
    particles_sat.z += 6.26 | units.kpc
    particles_sat.vx += 29.27 | units.kms
    particles_sat.vy += -37.54 | units.kms
    particles_sat.vz += -580.96 | units.kms

    gravity = Gadget2(converter_sat , number_of_workers=16)
    gravity.parameters.epsilon_squared = (0.1 | units.kpc)**2
    gravity.parameters.max_size_timestep = dt
    gravity.parameters.time_max = time_max
    set1 = gravity.particles.add_particles(particles_gal)
    set2 = gravity.particles.add_particles(particles_sat)

    sat_stars = gravity.particles.copy()[-int(1e5):]
    amuse.io.write_set_to_file(sat_stars, _DATADIR+'snap_inital.csv', format='csv')
   

    i = 0
    while gravity.model_time < tend:
        i = i+1
        gravity.evolve_model(i * dt)
        print(
            "evolved to:",
            gravity.model_time.as_quantity_in(units.Myr),
            "/",
            tend
        )
        if i % 10 == 0:
            sat_stars = gravity.particles.copy()[-int(2e5):]
            amuse.io.write_set_to_file(sat_stars, _DATADIR+'snap_%i.csv'%(i), format='csv')

    sat_stars = gravity.particles.copy()[-int(2e5):]
    amuse.io.write_set_to_file(sat_stars, _DATADIR+'snap_final.csv', format='csv')
        
    gravity.stop()

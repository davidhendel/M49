import numpy as np
import scipy
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from amuse.lab import *
from amuse.couple import bridge
from amuse.support.literature import LiteratureReferencesMixIn
from amuse.datamodel import Particles
from amuse.community.halogen.interface import Halogen
from amuse.community.gadget2.interface import Gadget2
import amuse.io
from galpy import potential
from galpy import orbit
from galpy.util import bovy_conversion
from galpy.potential import plotPotentials

def sq_softening(N, Rcluster):
    #nn = stars.nearest_neighbour()
    #if len(stars.x)==100000: return 0.066787233478082894 | units.parsec
    #else: return (np.median(np.sqrt((stars.x - nn.x)**2+(stars.y - nn.y)**2. + (stars.z - nn.z)**2.))**2.)
    epsilon = (N/Rcluster**3)**(-1/3)
    epsilon.unit = epsilon.unit.to_simple_form()
    return epsilon**2

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
Contributor Environment Setup (Part 2, Centos)
==============================================

The current CMakeList.txt assumes access to base directory. Please update to a suitable path:

.. code-block:: txt

    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT ./build/lib.linux-x86_64-3.7/horovod/)
    endif()

By default, if openMPI is not installed, Horovod will build with gloo. This can be confirmed. If this is good enough the below can be ignored.

.. code-block:: bash

    $ horovodrun --check-build

If you want to test your code with MPI, there are several issues with the open-mpi <= 3.1.3. Below are the instructions to update to 4.1.

.. code-block:: bash

    $ wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
    $ tar -xzf openmpi-4.1.1.tar.gz
    $ cd openmpi-4.1.1
    $ ./configure --prefix=/usr/lib64/openmpi
    $ make
    $ sudo make install
    $ module load mpi
    $ echo "export PATH=\$PATH:/usr/lib64/openmpi/bin" >> $HOME/.bashrc
    $ echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/opt/openmpi/lib" \ >> $HOME/.bashrc

If successful, this should be the result

.. code-block:: bash

    $ mpirun --version
    mpirun (Open MPI) 4.1.1

    Report bugs to http://www.open-mpi.org/community/help/

MPI 4 requires `GCC 8.2.0 <https://docs.hpc.shef.ac.uk/en/latest/sharc/software/parallel/openmpi-gcc.html>`__.
Refer to this `link <https://stackoverflow.com/questions/55345373/how-to-install-gcc-g-8-on-centos>`__ for installation instructions.

Now you can build Horovod:

.. code-block:: bash

    $ HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip install -v -e .

subroutine cgc(r, chi, f, box, sigma, kappa, N, energy)
    ! Compute Gaussian code interaction forces
    !
    ! Parameters
    ! ---------
    ! f : (N,D) numpy.ndarray
    !     Forces for N particles in D dimensions. Changed in place.
    ! r : (N,D) numpy.ndarray
    !     Positions for N particles in D dimensions.
    ! chi : (T,T) numpy.ndarray
    !     Interaction mixing energy between T species.
    ! box : (D,) numpy.ndarray
    !     D-dimensional simulation box size.
    ! kappa : float
    !     Incompressibility.
    ! sigma : float
    !     Filter width.
    ! N : int
    !     Total number of particles.
    !
    ! Returns
    ! -------
    ! energy : float
    !     Total energy of all two-particle bonds.
    !
    implicit none

    real(4), dimension(:,:), intent(in)     :: r
    real(4), dimension(:,:), intent(in)     :: chi
    real(4), dimension(:,:), intent(in out) :: f
    real(4), dimension(:),   intent(in)     :: box
    real(4),                 intent(in)     :: sigma
    real(4),                 intent(in)     :: kappa
    integer,                 intent(in)     :: N
    real(4),                 intent(out)    :: energy

    integer :: i, j
    real(4) :: rij2, rij_x, rij_y, rij_z
    real(4) :: denom, factor, phi0, V, pi, pi32, sigma5, sigma3
    real(4) :: e, efactor, df
    real(4) :: bx, by, bz

    pi = acos(-1.0)
    pi32 = sqrt(pi * pi * pi)
    denom = 1.0 / (4.0 * sigma * sigma)
    V = box(1) * box(2) * box(3)
    phi0 = N / V
    sigma3 = sigma * sigma * sigma
    sigma5 = sigma3 * sigma * sigma
    factor = 1.0 / (16.0 * pi32 * kappa * sigma5 * phi0)
    efactor = 1.0 / (8.0 * pi32 * kappa * sigma3 * phi0)

    energy = 0.0
    f = 0.0 ! Set all array elements

    bx = 1.0 / box(1)
    by = 1.0 / box(2)
    bz = 1.0 / box(3)

    do i = 1, N
      do j = i + 1, N
        rij_x = r(j, 1) - r(i, 1)
        rij_x = rij_x - box(1) * nint(rij_x * bx)

        rij_y = r(j, 2) - r(i, 2)
        rij_y = rij_y - box(2) * nint(rij_y * by)

        rij_z = r(j, 3) - r(i, 3)
        rij_z = rij_z - box(3) * nint(rij_z * bz)

        ! rij = sqrt(rij_x * rij_x + rij_y * rij_y + rij_z * rij_z)
        rij2 = rij_x * rij_x + rij_y * rij_y + rij_z * rij_z

        e = exp(- rij2 * denom) * (1.0 + kappa * chi(i, j))
        df = e * factor

        f(i, 1) = f(i, 1) - df * rij_x
        f(j, 1) = f(j, 1) + df * rij_x

        f(i, 2) = f(i, 2) - df * rij_y
        f(j, 2) = f(j, 2) + df * rij_y

        f(i, 3) = f(i, 3) - df * rij_z
        f(j, 3) = f(j, 3) + df * rij_z

        energy = energy + e * efactor
      end do
    end do
    energy = energy + (0.5 * N * efactor - 0.5 * N / kappa)
end subroutine

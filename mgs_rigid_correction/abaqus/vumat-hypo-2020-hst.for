      subroutine vumat(
      ! Read only (unmodifiable) variables
     &  nblock,ndir,nshr,nstatev,nfieldv,nprops,lanneal,stepTime,
     &  totalTime,dt,cmname,coordMp,charLength,props,density,strainInc,
     &  relSpinInc,tempOld,stretchOld,defgradOld,fieldOld,stressOld,
     &  stateOld,enerInternOld,enerInelasOld,tempNew,stretchNew,
     &  defgradNew,fieldNew,
      ! Write only (modifiable) variables
     &  stressNew,stateNew,enerInternNew,enerInelasNew)
      ! ================================================================
      ! Abaqus/Explicit user subroutine 'vumat' with implemented rate-
      ! independent hypoplastic model for non-cohesive soils in the 
      ! reference version (Wolffersdorff 1996) and the extended version 
      ! (Niemunis and Herle 1997). The reference model is available by 
      ! choosing the material constants m_R = m_T = 1. This user 
      ! subroutine is explained in Kelm (2004).
      ! 
      ! The reference model is a rate-independent endomorphous hypo-
      ! plastic model with the internal state variable $e$ (void ratio).
      ! The extended model includes a smooth transition from hypoelas-
      ! ticity with small strain stiffness to hypoplasticity and 
      ! includes the internal state variables $e$ (void ratio) and $\hb$
      ! (intergranular strain tensor). 
      !
      ! ----------------------------------------------------------------
      ! LITERATURE
      ! 
      ! If you use this user subroutine, don't forget to cite the 
      ! following publications:
      !
      !   Wolffersdorff P.-A. (1996): A hypoplastic relation for granular 
      !   materials with a predefined limit state surface. In: Mechanics 
      !   of Cohesive-frictional Materials 1(3):251-271, DOI:  
      !   10.1002/(SICI)1099-1484(199607)1:3<251::AID-CFM13>3.0.CO;2-3
      !
      !   Niemunis A., Herle I. (1997): Hypoplastic model for cohesionless 
      !   soils with elastic strain range. In: Mechanics of Cohesive-fric-
      !   tional Materials 2(4):279-299, DOI: 
      !   10.1002/(SICI)1099-1484(199710)2:4<279::AID-CFM29>3.0.CO;2-8
      !
      !   Kelm M. (2004): Numerische Simulation der Verdichtung rolliger 
      !   Böden mittels Vibrationswalzen. Dissertation. Veröffentlichungen
      !   des Instituts für Geotechnik und Baubetrieb, Heft 6
      !
      ! ----------------------------------------------------------------
      ! LICENSE
      ! 
      ! This user subroutine is non-free! Authorized persons are allowed
      ! to apply this software for scientific purposes only. The transfer
      ! of this software to other persons is not allowed.
      ! 
      ! ----------------------------------------------------------------
      ! PROGRAMMING
      !
      ! Original user subroutine 'umat.f' by Konrad Nübel in corporation
      ! with Andrzej Niemunis(KIT)
      !  - last change: 1999
      ! Migration from 'umat' to 'vumat' by Martin Kelm (TUHH)
      !  - last change: 23.04.2002
      ! Minor revision by Klaus-Peter Mahutka (TUHH)
      !  - last change: 24.03.2004
      !  - no calculation without intergranular strain possible because 
      !    not reasonable in dynamic simulations
      !  - no implicit integration of ige possible
      !  - calculation of the Jacobian just for small elasticity
      ! Minor revision by Tim Pucker (TUHH)
      !  - last change 25.11.2010
      !  - inserted a minimum stiffness for linear elasticity to avoid 
      !    numerical problems in Abaqus/Explicit CEL simulations
      ! Minor revision by Hans Stanford (TUHH)
      !  - usage of more Fortran 90/95 standard
      !  - controlled program stop
      !  - comments added
      !  - last change 09.01.2020
      !
      ! ----------------------------------------------------------------
      ! ABAQUS INPUT FILE
      !
      !  *solid section,elset=soil,material=sand
      !  *material,name=sand
      !  *density
      !   rho_d
      !  *depvar
      !   20
      !  *user material,unsymm,const=16 
      !   phi_c, p_c, h_s, en, e_d0, e_c0, e_i0, alpha
      !   beta, m_T, m_R, Rmax, betax, Chi, , e_0
      !
      ! Don't use the capillary pressure $p_c$, its implementation is 
      ! not proofed yet. You can specify the initial void ratio $e_0$
      ! but it will not be recognized in this subroutine. Define $e_0$
      ! as an initial condition as explained above.
      ! 
      ! Don't forget to specify the initial conditions for all state 
      ! variables as stress tensor, void ratio, and intergranular strain
      ! tensor either in the input file:
      ! 
      !  *initial conditions,type=stress,geostatic
      !   element/elset, sigma'_z1, z1, sigma'_z2, z2, K_0x, K_0y 
      !  *initial conditions,type=solution
      !   elset, e_0, , , , , , h_11 
      !   h_22, h_33, h_12, h_13, h_23
      ! 
      ! or via user subroutine 'sigini' (initial stresses) and 'sdvini'
      ! (internal state variables). Invoke this user subroutines in the 
      ! input file as follows:
      ! 
      !  *initial conditions,type=stress,user
      !  *initial conditions,type=solution
      ! 
      ! The initial stresses and the initial void ratio are mandatory. 
      ! The initial intergranular strains depends on the strain history
      ! and can be zero or -Rmax. See Niemunis and Herle (1997) for 
      ! details.
      !
      ! ----------------------------------------------------------------
      ! STATE VARIABLES
      !
      !   stateOld(1)  ... current void ratio $e$
      !   stateOld(2)  ... volumetric strain increment $\tr\Db$
      !   stateOld(3)  ... mean pressure $p=-\tr\Tb/3$
      !   stateOld(4)  ... deviatoric stress $q=\sqrt(\frac{3}{2}||
      !                    \dev(\Tb)||)$
      !   stateOld(5)  ... mobilized friction angle $\sin(\varphi_{mob}$)
      !   stateOld(6)  ... dto. $\arctan(\frac{\sigma_{12}}{\sigma_{22}})$
      !   stateOld(7-12) . intergranular strain tensor $\hb_{11}$, 
      !                    $\hb_{22}$, $\hb_{33}$, $\hb_{12}$, 
      !                    $\hb_{13}$, $\hb_{23}$
      !   stateOld(13) ... capillary pressure $p_c$
      !   stateOld(14) ... Euclidean norm of Jaumann rate of stress 
      !                    tensor $||\kTb||_{old}$
      !   stateOld(15) ... tension cut-off flag (1=soil, 2=tension cut-off)
      !   stateOld(16) ... initial void ratio $e_0$ 
      !   stateOld(17) ... compaction $e_0-e$
      !   stateOld(18) ... stiffness ddsdde(1,1)
      !   stateOld(19) ... rho
      !   stateOld(20) ... free
      !
      ! ================================================================
      ! IMPORTANT HINTS
      ! 
      !  - Look for 'Control parameters (****)' in this subroutine and 
      !    check default values. Change this default values only, if you
      !    are sure what you do.
      !
      !  - Use Abaqus/Explicit in the double precision mode only. Please
      !    check this in the Abaqus environment file abaqus_v6.env 
      !    (double precision=both) or specify it as a command line 
      !    option (abaqus job=... inp=... user=... double=both).
      !
      !  - Ensure that Abaqus uses the initial void ratio you have 
      !    specified.
      !
      ! ================================================================
      include 'vaba_param.inc'
      dimension props(nprops), density(nblock), coordMp(nblock,*),
     &  charLength(nblock), strainInc(nblock, ndir+nshr),
     &  relSpinInc(nblock,nshr), tempOld(nblock),
     &  stretchOld(nblock,ndir+nshr),
     &  defGradOld(nblock, ndir+nshr+nshr),
     &  fieldOld(nblock,nfieldv), stressOld(nblock,ndir+nshr),
     &  stateOld(nblock,nstatev), enerInternOld(nblock),
     &  enerInelasOld(nblock), tempNew(nblock),
     &  stretchNew(nblock,ndir+nshr),
     &  defgradNew(nblock,ndir+nshr+nshr),
     &  stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev),
     &  enerInternNew(nblock), enerInelasNew(nblock)
      character(len=80) :: cmname
      ! Local variables
      integer :: i,j,k,l,ij,nstep,istep,kl,iescape,loading,i1,i2,i3
      integer :: nbase(3,3),nbaseM(3,3),i6(6),j6(6)
      real(kind=8) :: zero,one,two,three,sq2,sq3,sq6,mu,sub,ela,phic,hs,
     &  en,ed0,ec0,ei0,alpha,beta,m_T,m_R,Rmax,betax,Chi,epor0,az,trD,
     &  trT,x,norm33D,dtajm,normqT,T11,T22,T33,trT_d2,trT_s2,trT_s3,
     &  normT_s,cos3th,tanbe,term1,term2,Fm,bauer,ed,ec,ei,fb,fe,fd,
     &  Fmsq,azsq,sstajm,norme3,rho,direction,ssdt,rhox,factor1,factor2,
     &  factor3,rhob2,aux1,rmohr,smohr,tmax,tmin,soft,ela2,ddsdde_min,
     &  ddsdde(9,9),ddsdde11
      real(kind=8) :: delta(3,3),T(6),D(6),e3_e3s(6),LLout(3,3,3,3),
     &  LLxx(3,3,3,3),LLtt(3,3,3,3),T_d(6),T_s(6),NN(3,3),LL(3,3,3,3),
     &  LD(3,3),T_dot(6),ssn(6),MM(3,3,3,3),NNnn(3,3,3,3),ssnn(3,3,3,3),
     &  T_dev(6),drot_t(3,3),eR_t(3,3),ReR_t(3,3),epor(nblock)
      integer :: ntens
      data nbase  /1,4,5,4,2,6,5,6,3/
      data nbaseM /1,4,5,7,2,6,7,7,3/
      data delta /1.d0,0.d0,0.0d0,0.0d0,1.0d0,0.0d0,0.0d0,0.0d0,1.0d0/
      data i6/1,2,3,1,1,2/, j6/1,2,3,2,3,3/
      parameter(zero=0.d0,one=1.d0,two=2.d0,three=3.d0)
      parameter(sq2=1.4142135623730951455d0,
     &  sq3=1.7320508075688771931d0,
     &  sq6=2.4494897427831778813d0)
      ! Control parameters (****)
      ! ... controls the magnitude of intergranular strain increment
      parameter(mu=0.10d0)
      ! ... maximum strain increment without substepping
      parameter(sub=1.d-05)
      ! ... tension cut-off limit stress level
      parameter(ela=-0.1d0)
      ! ... controls assembling of stiffness matrix
      parameter(ela2=-1.0d0)
      ! ... minimum Young's modulus for linear elasticity
      parameter(ddsdde_min=1.0d0)
 
      ! Check time increment 
      if (dt <= zero) dt = one
      ! Material constants 
      phic = props(1)
      hs = props(3)
      en = props(4)
      ed0 = props(5)
      ec0 = props(6)
      ei0 = props(7)
      alpha = props(8)
      beta = props(9)
      m_T = props(10)
      m_R = props(11)
      Rmax = props(12)
      betax = props(13)
      Chi = props(14)
      epor0 = props(16)
      az = sq3*(three-dsin(phic))/(two*sq2*dsin(phic))
      ! Initialize state variables
      if (stepTime == zero) then     
        do km=1,nblock
          stateNew(km,13) = zero
          epor(km) = stateOld(km,1)
          stateNew(km,16) = stateOld(km,16)
        enddo      
      endif
      ! For first increment initialize void ratio, if missing      
      if (stepTime == dt) then      
        ! first increment
        do km=1,nblock
          ! capillary pressure via void ratio
          stateNew(km,13) = max(stateOld(km,13),props(2)) 
          trT = zero
          trT = stressOld(km,1)+stressOld(km,2)+stressOld(km,3)
          if (stateOld(km,1) <= 0.001) then
            epor(km) = epor0*exp(-(-trT/hs)**en)
            stateNew(km,16) = epor(km)
          else
            epor(km) = stateOld(km,1)
            stateNew(km,16) = stateOld(km,16)
          endif
        enddo
      else
        ! subsequent increments
        do km=1,nblock
          epor(km) = stateOld(km,1)
        enddo
      endif
      ! Main loop over material points
      main_loop: do km=1,nblock     
        trD = zero
        trT = zero
        T = zero
        D = zero
        e3_e3s = zero
        do i=1,ndir
          T(i) = stressOld(km,i)
          D(i) = strainInc(km,i)/dt
          e3_e3s(i) = stateOld(km,6+i)
          trD = trD+D(i)
          trT = trT+T(i)
        enddo
        ! Change from Abaqus/Standard to Abaqus/Standard storage 
        ! scheme for symmetric tensors: {s12,s23,s31} -> {s12,s13,s23}
        T(4) = stressOld(km,4)
        D(4) = strainInc(km,4)/dt
        e3_e3s(4) = 0.5d0*stateOld(km,6+4)
        if (nshr > 1) then      
          T(5) = stressOld(km,6)
          D(5) = strainInc(km,6)/dt
          e3_e3s(5) = 0.5d0*stateOld(km,6+5)
          T(6) = stressOld(km,5)
          D(6) = strainInc(km,5)/dt
          e3_e3s(6)= 0.5d0*stateOld(km,6+6)
        endif
        ! calculate stress increment 'dstress' by using  forward Euler 
        ! with small-strain substepping and using  backward Euler for 
        ! intergranular strain subsubstepping
        norm33D = sqrt(D(1)*D(1)+D(2)*D(2)+D(3)*D(3)
     &    +two*(D(4)*D(4)+D(5)*D(5)+D(6)*D(6)))
        ! subincrements if ||D|| > sub (!=small-strain substeps)
        ! if we are at the beginning no substepping
        if (stepTime <= zero) then
          nstep = 1
          dtajm = dt
        else
          nstep = max(idnint(one/sub*norm33D*dt), 1)
          dtajm = dt/nstep
        endif
        ! Start substepping loop
        substepping_loop: do istep=1,nstep
          sstajm = zero
          ! Begin of subsubstepping loop for intergranular strain
202       continue
          stateNew(km,15) = one ! hypoplastic/elastic flag
          trT = T(1)+T(2)+T(3) 
          tension_cutoff_check: if (trT > ela) then
            ! Linear elastic model with small stiffness (tension
            ! cut-off), if tensile stresses occur
            LLout = zero
            ddsdde11 = stateOld(km,18)        
            call k_elasticity(LLtt,T,D,dtajm,nbaseM,ddsdde11,ddsdde_min)
            call k_update_all(LLout,LLtt,dtajm/dt)
            e3_e3s(1:6) = zero
            stateNew(km,15) = two
            ! assemble jacobian matrix (secant stiffness)
            do i=1,ndir+nshr
              do j=1,ndir+nshr
                if (j <= ndir) ddsdde(i,j) =
     &            LLout(i6(i),j6(i),i6(j),j6(j))
                if (j > ndir) ddsdde(i,j) =  0.5d0*
     &            (LLout(i6(i),j6(i),i6(j),j6(j))+
     &            LLout(i6(i),j6(i),j6(j),i6(j)))
              enddo
            enddo
            if (strainInc(km,1) == zero) then
              stateNew(km,18) = zero
            else
              if (strainInc(km,2) == zero) then
              else
                if (strainInc(km,3) == zero) then
                else
                 stateNew(km,18) = abs((
     &             (stressNew(km,1)-stressOld(km,1))/strainInc(km,1)+
     &             (stressNew(km,2)-stressOld(km,2))/strainInc(km,2)+
     &             (stressNew(km,3)-stressOld(km,3))/strainInc(km,3))/3.d0)
                endif
              endif
            endif
          else 
            ! regular case, hypoplastic model
            normqT = T(1)*T(1)+T(2)*T(2)+T(3)*T(3)+
     &        two*(T(4)*T(4)+T(5)*T(5)+T(6)*T(6))
            T_d(1:6) = T(1:6)/trT
            T_s(1:6) = T_d(1:6)
            T_dev(1:6) = T(1:6)
            T_s(1:3) = T_s(1:3)-one/three
            T_dev(1:3) = T_dev(1:3)-one/three
            T11 = T(1)-trT/three
            T22 = T(2)-trT/three
            T33 = T(3)-trT/three
            trT_d2 = normqT/(trT*trT)
            trT_s2 = (T11*T11+T22*T22+T33*T33+
     &        two*(T(4)*T(4)+T(5)*T(5)+T(6)*T(6)))/(trT*trT)
            trT_s3 = (T11*T11*T11+T22*T22*T22+T33*T33*T33+
     &        T11*three*(T(4)*T(4)+T(5)*T(5))+
     &        T22*three*(T(4)*T(4)+T(6)*T(6))+
     &        T33*three*(T(5)*T(5)+T(6)*T(6))+
     &        6.d0*(T(4)*T(5)*T(6)))/(trT*trT*trT)
            normT_s = sqrt(trT_s2)
            ! scalar function Fm
            cos3th = one
            if (trT_s2 > 1.d-10) cos3th = -sq6*trT_s3/(trT_s2**1.5d0)
            if (cos3th > one) cos3th = one
            if (cos3th < -one) cos3th = -one
            tanbe = sq3*normT_s
            term1 = two+sq2*tanbe*cos3th
            term2 = two-tanbe*tanbe
            Fm = sqrt(abs(tanbe*tanbe/8.0d0+term2/term1))-
     &        sq2*tanbe*0.25d0
            ! limiting void ratios
            Bauer = exp(-(-trT/hs)**en)
            ed = ed0*Bauer
            ec = ec0*Bauer
            ei = ei0*Bauer
            if (epor(km) > ei) epor(km) = ei
            if (epor(km) < ed) epor(km) = ed
            ! barotropy and pycnotropy factors 
            term1 = three+az*az-az*sq3*((ei0-ed0)/(ec0-ed0))**alpha
            if (term1 < zero) then
              write(*,*) 'VUMAT ERROR: factor fb not defined'
              call xplb_exit
            endif
            fb = hs/en/term1*(one+ei)/ei*(ei0/ec0)**beta*(-trT/hs)**(one-en)
            fe = (ec/epor(km))**beta
            fd = ((epor(km)-ed)/(ec-ed))**alpha
            ! tensor function L and N
            term1 = fb*fe/trT_d2
            term2 = term1*fd*Fm*az
            Fmsq = Fm*Fm
            azsq = az*az
            do i=1,3
              do j=1,3
                ij = nbase(i,j)
                NN(i,j) = term2*(T_d(ij)+T_s(ij))
                do k=1,3
                  do l=1,3
                     kl = nbase(k,l)
                     LL(i,j,k,l) = term1*(Fmsq*delta(i,k)*delta(j,l)+
     &                 azsq*T_d(kl)*T_d(ij))
                  enddo
                enddo
              enddo
            enddo           
            ! Jaumann rate of stress tensor T_dot=MD if intergranular 
            ! strain is activated; findout the direction
            norme3 = sqrt(e3_e3s(1)*e3_e3s(1)+e3_e3s(2)*e3_e3s(2)+
     &        e3_e3s(3)*e3_e3s(3)+two*(e3_e3s(4)*e3_e3s(4)+
     &        e3_e3s(5)*e3_e3s(5)+e3_e3s(6)*e3_e3s(6)))
            rho = max(norme3/Rmax, 1.0d-10)
            direction = e3_e3s(1)*D(1)+e3_e3s(2)*D(2)+e3_e3s(3)*D(3)+
     &        two*(e3_e3s(4)*D(4)+e3_e3s(5)*D(5)+e3_e3s(6)*D(6))
            ! calculate new subsubincrement ssdt at strain reversals
            ! and a flag iescape:=1 if it is the last sub-subincrement
            ! in a given subincrement
            call k_get_ssdt(Rmax,betax,mu,rho,norme3,norm33D,direction,
     &        dtajm,sstajm,iescape,ssdt)
            ! Correction of stiffness for small-strain effects
            ! calculate
            !   ssn  =  delta_hat
            !   ssnn =  delta_hat*delta_hat
            !   NNnn =  N*delta_hat
            !   LLxx =  L:(delta_hat*delta_hat)
            ssn(1:6) = e3_e3s(1:6)/(rho*Rmax)
            do i=1,3
              do j=1,3
                ij = nbase(i,j)
                do k=1,3
                  do l=1,3
                    kl = nbase(k,l)
                    ssnn(i,j,k,l) = ssn(ij)*ssn(kl)
                    NNnn(i,j,k,l) = NN(i,j)*ssn(kl)
                  enddo
                enddo
              enddo
            enddo
            do i=1,3
              do j=1,3
                do k=1,3
                  do l=1,3
                    LLxx(i,j,k,l) = LL(i,j,1,1)*ssnn(1,1,k,l)+
     &                              LL(i,j,1,2)*ssnn(1,2,k,l)+
     &                              LL(i,j,1,3)*ssnn(1,3,k,l)+
     &                              LL(i,j,2,1)*ssnn(2,1,k,l)+
     &                              LL(i,j,2,2)*ssnn(2,2,k,l)+
     &                              LL(i,j,2,3)*ssnn(2,3,k,l)+
     &                              LL(i,j,3,1)*ssnn(3,1,k,l)+
     &                              LL(i,j,3,2)*ssnn(3,2,k,l)+
     &                              LL(i,j,3,3)*ssnn(3,3,k,l)
                  enddo
                enddo
              enddo
            enddo
            ! soft transition from hypoe4lasticity to hypoplasticity
            if (rho>one) then
              soft = one
            else
              soft = exp(-(one-rho)/0.05d0)
            endif
            rhox = rho**Chi
            factor1 = rhox*m_T+(one-rhox)*m_R
            factor2 = rhox*(one-m_T)
            factor3 = rhox*(m_R+(one-m_R)*soft-m_T)
            if (direction > zero) then
              loading = 1
              do i=1,3
                do j=1,3
                  do k=1,3
                    do l=1,3
                      MM(i,j,k,l) = factor1*LL(i,j,k,l)+
     &                 factor2*LLxx(i,j,k,l)+rhox*NNnn(i,j,k,l)
                    enddo
                  enddo
                enddo
              enddo
            else
              loading = 0
              do i=1,3
                do j=1,3
                  do k=1,3
                    do l=1,3
                      MM(i,j,k,l) = factor1*LL(i,j,k,l)+
     &                  factor3*LLxx(i,j,k,l)+rhox*NNnn(i,j,k,l)*soft
                    enddo
                  enddo
                enddo
              enddo
            endif
            ! stress rate
            do i=1,3
              do j=1,3
                 ij = nbaseM(i,j)
                if(ij<=6) then
                   T_dot(ij) = MM(i,j,1,1)*D(1)+
     &                         MM(i,j,1,2)*D(4)+
     &                         MM(i,j,1,3)*D(5)+
     &                         MM(i,j,2,1)*D(4)+
     &                         MM(i,j,2,2)*D(2)+
     &                         MM(i,j,2,3)*D(6)+
     &                         MM(i,j,3,1)*D(5)+
     &                         MM(i,j,3,2)*D(6)+
     &                         MM(i,j,3,3)*D(3)
                endif
              enddo
            enddo
            ! update of state variables 
            ! use explicit 
            if (loading == 1) then
              ! loading
              rhob2 = zero
              if (norme3 > 1.d-10)
     &          rhob2 = direction*((norme3/Rmax)**betax)/(norme3*norme3)
              do i=1,6
                e3_e3s(i) = e3_e3s(i)+D(i)*ssdt-e3_e3s(i)*rhob2*ssdt
              enddo
              norme3 = sqrt(e3_e3s(1)*e3_e3s(1)+e3_e3s(2)*e3_e3s(2)+
     &          e3_e3s(3)*e3_e3s(3)+
     &          two*(e3_e3s(4)*e3_e3s(4)+e3_e3s(5)*e3_e3s(5)+
     &          e3_e3s(6)*e3_e3s(6)))
              stateNew(km,19) = norme3*1.d06
              if (norme3 > Rmax) then
                ! correction of intergranular strain
                factor1 = (norme3-Rmax)/norme3
                do i=1,6
                  e3_e3s(i) = e3_e3s(i)-e3_e3s(i)*factor1
                enddo
              endif
            else
              ! unloading
              do i=1,6
                e3_e3s(i) = e3_e3s(i)+D(i)*ssdt
              enddo
              stateNew(km,19) = norme3*1.d06
            endif
            T(1:6) = T(1:6)+T_dot(1:6)*ssdt
            aux1 = two*ssdt*((istep-1)*dtajm+sstajm+0.5d0*ssdt)/(dt*dt)
            sstajm = sstajm+ssdt
            ! update void ratio 
            epor(km) = epor(km) +(one+epor(km))*trD*ssdt
            ! if the stress near to tension assemble Jacobian for 
            ! linear elasticity with small stiffness
            if (trT > ela2) then                       
              LLout(1:3,1:3,1:3,1:3) = zero
              call k_tangent_stiffness_ige(LLout,MM)
              e3_e3s(1:6) = zero
              ! assemble jacobian matrix (secant stiffness)
              do i=1,ndir+nshr
                do j=1,ndir+nshr
                  if (j <= ndir) ddsdde(i,j) =
     &              LLout(i6(i),j6(i),i6(j),j6(j))
                  if (j > ndir) ddsdde(i,j) = 0.5d0*
     &             (LLout(i6(i),j6(i),i6(j),j6(j))
     &             +LLout(i6(i),j6(i),j6(j),i6(j)))
                enddo
              enddo
              if (strainInc(km,1) == zero) then
                stateNew(km,18) = zero
              else
                if (strainInc(km,2) == zero) then
                else
                  if (strainInc(km,3) == zero) then
                  else
                   stateNew(km,18) = abs(
     &               ((stressNew(km,1)-stressOld(km,1))/strainInc(km,1)+
     &                (stressNew(km,2)-stressOld(km,2))/strainInc(km,2)+
     &                (stressNew(km,3)-stressOld(km,3))/strainInc(km,3))/3.d0)
                  endif
                endif
              endif
            endif
            ! return to begin of subsubstepping loop, if necessary
            if (iescape /= 1) goto 202
            continue
          endif tension_cutoff_check
        enddo substepping_loop
        ! Update of state variables
        stressNew(km,1) = T(1)
        stressNew(km,2) = T(2)
        stressNew(km,3) = T(3)
        stateNew(km,6+1) = e3_e3s(1)
        stateNew(km,6+2) = e3_e3s(2)
        stateNew(km,6+3) = e3_e3s(3)
        ! Change back from Abaqus/Standard to Abaqus/Explicit storage 
        ! scheme for symmetric tensors: {s12,s13,s23} -> {s12,s23,s31}
        stressNew(km,4) = T(4)
        stateNew(km,6+4) = two*e3_e3s(4)
        if (nshr > 1) then
          stressNew(km,5) = T(6)
          stateNew(km,6+5) = two*e3_e3s(5)
          stressNew(km,6) = T(5)
          stateNew(km,6+6) = two*e3_e3s(6)
        endif
        stateNew(km,1) = epor(km)
        stateNew(km,2) = trD*dt
        if (strainInc(km,1) == zero) then
          stateNew(km,18) = zero
        else
          if (strainInc(km,2) == zero) then
          else
            if (strainInc(km,3) == zero) then
            else
             stateNew(km,18) = abs(
     &         ((stressNew(km,1)-stressOld(km,1))/strainInc(km,1)+
     &          (stressNew(km,2)-stressOld(km,2))/strainInc(km,2)+
     &          (stressNew(km,3)-stressOld(km,3))/strainInc(km,3))/3.d0)
            endif
          endif
        endif
        ! Abaqus output variables if required
        stateNew(km,3)= -(T(1)+T(2)+T(3))/three
        if (stateNew(km,15) > 1.5d0) then
          ! tension cut-off (linear elasticity)
          stateNew(km,4) = 0.0d0
          stateNew(km,5) = 0.0d0
          stateNew(km,6) = 0.0d0
          if (stepTime > dt) stateNew(km,16) = stateOld(km,16)
          stateNew(km,17) = stateOld(km,17)
        else
          ! regular case (hypoplasticity)
          Rmohr = sqrt((T(1)-T(2))*(T(1)-T(2))/4.d0+T(4)*T(4))
          Smohr = (T(1)+T(2))/two
          tmax = max(T(3), Smohr+Rmohr)
          tmin = min(T(3), Smohr-Rmohr)
          stateNew(km,4) = tmax-tmin
          stateNew(km,5) = 1
          stateNew(km,6) = atan(T(4)/T(2))
          if (stepTime > dt) stateNew(km,16) = stateOld(km,16)
          stateNew(km,17) = epor(km)-stateOld(km,16)
        endif
      enddo main_loop
      end subroutine vumat

      subroutine k_elasticity(dj,T,D,dtajm,nbaseM,ddsdde11,ddsdde_min)
      ! ----------------------------------------------------------------
      ! Returns stiffness and stress rate for linear elastic model in
      ! case of tension cut-off. Variables:
      !   dj ........ Jacobian
      !   ddsdde11 .. gradient last step,cmp stateNew(km,18)
      !   T ......... stress vector
      !   D ......... strain rate vector
      !   dtajm ..... sub time increment
      !   nbaseM .... mapping matrix
      ! ----------------------------------------------------------------
      implicit none
      integer, intent(in) :: nbaseM(3,3)
      real(kind=8), intent(in) :: ddsdde11,D(6),dtajm,ddsdde_min
      real(kind=8), intent(inout) :: T(6)
      real(kind=8), intent(out) :: dj(3,3,3,3)
      ! Local variables
      integer :: i,j,ij,k,l
      real(kind=8) :: T_dot(6),zero,half,E,help
      parameter(zero=0.0d0,half=0.5d0)
      
      E = ddsdde11
      if (E < ddsdde_min) E = ddsdde_min
      dj(1:3,1:3,1:3,1:3) = zero
      dj(1,1,1,1) = E
      dj(2,2,2,2) = E
      dj(3,3,3,3) = E
      help = E*half
      dj(1,2,1,2) = help
      dj(1,2,2,1) = help
      dj(2,1,1,2) = help
      dj(2,1,2,1) = help
      dj(1,3,1,3) = help
      dj(1,3,3,1) = help
      dj(3,1,1,3) = help
      dj(3,1,3,1) = help
      dj(2,3,2,3) = help
      dj(2,3,3,2) = help
      dj(3,2,2,3) = help
      dj(3,2,3,2) = help
      do i=1,3
        do j=1,3
          ij = nbaseM(i,j)
          if (ij <= 6) then
          T_dot(ij) = dj(i,j,1,1)*D(1)+
     &                dj(i,j,1,2)*D(4)+
     &                dj(i,j,1,3)*D(5)+
     &                dj(i,j,2,1)*D(4)+
     &                dj(i,j,2,2)*D(2)+
     &                dj(i,j,2,3)*D(6)+
     &                dj(i,j,3,1)*D(5)+
     &                dj(i,j,3,2)*D(6)+
     &                dj(i,j,3,3)*D(3)
          endif
        enddo
      enddo
      ! Update stress
      T(1:3) = T(1:3)+T_dot(1:3)*dtajm
      T(4:6) = zero
      end subroutine k_elasticity

      subroutine k_update_all(LLout,LLtt,dtime)
      ! ---------------------------------------------------------------
      ! Update of secant stiffness.
      ! ---------------------------------------------------------------
      implicit none
      real(kind=8), intent(in) :: LLtt(3,3,3,3),dtime
      real(kind=8), intent(inout) :: LLout(3,3,3,3)

      LLout(1,1,1,1) = LLout(1,1,1,1)+LLtt(1,1,1,1)*dtime
      LLout(2,2,2,2) = LLout(2,2,2,2)+LLtt(2,2,2,2)*dtime
      LLout(3,3,3,3) = LLout(3,3,3,3)+LLtt(3,3,3,3)*dtime
       
      LLout(1,2,1,2) = LLout(1,2,1,2)+LLtt(1,2,1,2)*dtime
      LLout(1,2,2,1) = LLout(1,2,2,1)+LLtt(1,2,2,1)*dtime
      LLout(2,1,1,2) = LLout(2,1,1,2)+LLtt(2,1,1,2)*dtime
      LLout(2,1,2,1) = LLout(2,1,2,1)+LLtt(2,1,2,1)*dtime
       
      LLout(1,3,1,3) = LLout(1,3,1,3)+LLtt(1,3,1,3)*dtime
      LLout(1,3,3,1) = LLout(1,3,3,1)+LLtt(1,3,3,1)*dtime
      LLout(3,1,1,3) = LLout(3,1,1,3)+LLtt(3,1,1,3)*dtime
      LLout(3,1,3,1) = LLout(3,1,3,1)+LLtt(3,1,3,1)*dtime
      
      LLout(2,3,2,3) = LLout(2,3,2,3)+LLtt(2,3,2,3)*dtime
      LLout(2,3,3,2) = LLout(2,3,3,2)+LLtt(2,3,3,2)*dtime
      LLout(3,2,2,3) = LLout(3,2,2,3)+LLtt(3,2,2,3)*dtime
      LLout(3,2,3,2) = LLout(3,2,3,2)+LLtt(3,2,3,2)*dtime
 
      LLout(1,1,2,2) = LLout(1,1,2,2)+LLtt(1,1,2,2)*dtime
      LLout(1,1,3,3) = LLout(1,1,3,3)+LLtt(1,1,3,3)*dtime
      LLout(2,2,3,3) = LLout(2,2,3,3)+LLtt(2,2,3,3)*dtime
      LLout(2,2,1,1) = LLout(2,2,1,1)+LLtt(2,2,1,1)*dtime
      LLout(3,3,1,1) = LLout(3,3,1,1)+LLtt(3,3,1,1)*dtime
      LLout(3,3,2,2) = LLout(3,3,2,2)+LLtt(3,3,2,2)*dtime
      end subroutine k_update_all

      subroutine k_tangent_stiffness_ige(LLout,MM)
      !----------------------------------------------------------------
      ! Update of tangent stiffnes if intergranular strain is used.
      !----------------------------------------------------------------
      implicit none
      real(kind=8), intent(in) :: MM(3,3,3,3)
      real(kind=8), intent(out) :: LLout(3,3,3,3)

      LLout(1,1,1,1) = MM(1,1,1,1)
      LLout(2,2,2,2) = MM(2,2,2,2)
      LLout(3,3,3,3) = MM(3,3,3,3)

      LLout(1,2,1,2) = MM(1,2,1,2)
      LLout(1,2,2,1) = MM(1,2,2,1)
      LLout(2,1,1,2) = MM(2,1,1,2)
      LLout(2,1,2,1) = MM(2,1,2,1)

      LLout(1,3,1,3) = MM(1,3,1,3)
      LLout(1,3,3,1) = MM(1,3,3,1)
      LLout(3,1,1,3) = MM(3,1,1,3)
      LLout(3,1,3,1) = MM(3,1,3,1)

      LLout(2,3,2,3) = MM(2,3,2,3)
      LLout(2,3,3,2) = MM(2,3,3,2)
      LLout(3,2,2,3) = MM(3,2,2,3)
      LLout(3,2,3,2) = MM(3,2,3,2)

      LLout(1,1,2,2) = MM(1,1,2,2)
      LLout(1,1,3,3) = MM(1,1,3,3)
      LLout(2,2,3,3) = MM(2,2,3,3)
      LLout(2,2,1,1) = MM(2,2,1,1)
      LLout(3,3,1,1) = MM(3,3,1,1)
      LLout(3,3,2,2) = MM(3,3,2,2)
      end subroutine k_tangent_stiffness_ige

      subroutine k_get_ssdt(Rmax,betax,mu,rho,norme3,norm33D,direction,
     &  dtajm,sstajm,iescape,ssdt)
      ! ----------------------------------------------------------------
      ! Computes a time subsubincrement ssdt necessary at strain rever-
      ! sals and a flag iescape = 1, if it is the last subsubincrement 
      ! in the current subincrement.
      ! ----------------------------------------------------------------
      implicit none
      real(kind=8), intent(in) :: Rmax,betax,mu,rho,norme3,norm33D,
     &  direction,dtajm,sstajm
      integer, intent(out) :: iescape
      real(kind=8), intent(inout) :: ssdt
      ! Local variables
      real(kind=8) :: ssdt_min
      
      ! If nothing to do (for example if norm33D=0)
      if (norm33D*dtajm <= mu*Rmax) then
        ssdt = dtajm
        iescape = 1
        return
      endif
      ssdt_min = mu*Rmax/norm33D
      ! Reduction of increment length as a function of rho
      if (direction > 0d0) then
        ! loading
        if (rho < 0.99d0) then
          ssdt = min(1.01d0*dtajm,abs(0.1d0*Rmax/
     &      ((1.0d0-rho**betax)*norm33D)))
        else
          ssdt = 1.01d0*dtajm
        endif
      else
        ! unloading
        ssdt = ssdt_min
      endif
      ! Reduction of increment length with respect to direction
      if (rho > 0.01d0 .and. direction >= 0.0d0) then
        ssdt = ssdt_min+(ssdt-ssdt_min)*direction/(norme3*norm33D)
      endif
      ! if this is the last subsubincrement
      if (sstajm+ssdt >= dtajm)  then
        ssdt=dtajm-sstajm
        iescape = 1
      else
        iescape = 0
      endif
      end subroutine k_get_ssdt

subroutine vumat(  &
     &  nblock, ndir, nshr, nstatev, nfieldv, nprops, jInfoArray, &
     &  stepTime, totalTime, dtArray, cmname, coordMp, charLength, &
     &  props, density, strainInc, relSpinInc, &
     &  tempOld, stretchOld, defgradOld, fieldOld, &
     &  stressOld, stateOld, enerInternOld, enerInelasOld, &
     &  tempNew, stretchNew, defgradNew, fieldNew, &
     &  stressNew, stateNew, enerInternNew, enerInelasNew ) 
      use iso_c_binding, only: c_char, c_null_char
      
      include 'vaba_param.inc'
      ! * the actual data format (single/double precision) is determined by the .inc file
      ! * the .inc file defines the implicit real kind as:
      ! * implicit real (a-h,o-z)
      ! * parameter (j_sys_Dimension = 1)
      ! * parameter( maxblk = 512 )
      ! use the implicit real kind
      ! Also include the .inc file in the interface to ensure consistent data type

      !!! TODO: we only support double precision in actual invoke_ calls.
      !!!       support both single and double precision in the future

      parameter (i_info_AnnealFlag = 1, &
          & i_info_Intpt    = 2, & ! Integration station number
          & i_info_layer  = 3, & ! Layer number
          & i_info_kspt   = 4, & ! Section point number in current layer
          & i_info_effModDefn = 5, & ! =1 if Bulk/ShearMod need to be defined
          & i_info_ElemNumStartLoc   = 6) ! Start loc of user element number

      dimension props(nprops), density(nblock), coordMp(nblock,*), &
     &  charLength(nblock), dtArray(2*(nblock)+1), strainInc(nblock,ndir+nshr), &
     &  relSpinInc(nblock,nshr), tempOld(nblock), &
     &  stretchOld(nblock,ndir+nshr), &
     &  defgradOld(nblock,ndir+nshr+nshr), &
     &  fieldOld(nblock,nfieldv), stressOld(nblock,ndir+nshr), &
     &  stateOld(nblock,nstatev), enerInternOld(nblock), &
     &  enerInelasOld(nblock), tempNew(nblock), &
     &  stretchNew(nblock,ndir+nshr), &
     &  defgradNew(nblock,ndir+nshr+nshr), &
     &  fieldNew(nblock,nfieldv), &
     &  stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev), &
     &  enerInternNew(nblock), enerInelasNew(nblock), jInfoArray(*)

      character*80 cmname

      pointer (ptrjElemNum, jElemNum)
      dimension jElemNum(nblock)

      interface
        function invoke_pt_vumat_batch( &
          & module_name, defgradF, nblock, ndir, nshr, &
          & par_mat, n_par_mat, enerInternNew, stressNew) result(err) bind(C)
            use iso_c_binding, only: c_char, c_int
            include 'vaba_param.inc'

            integer(c_int) :: err
            character(c_char), dimension(*), intent(in) :: module_name
            integer(c_int), value,           intent(in) :: nblock
            integer(c_int), value,           intent(in) :: ndir
            integer(c_int), value,           intent(in) :: nshr
            integer(c_int), value,           intent(in) :: n_par_mat
            dimension defgradF(*)
            dimension par_mat(*)
            dimension enerInternNew(*)
            dimension stressNew(*)
        end function invoke_pt_vumat_batch
      end interface

      integer :: i, k, err
      real(kind(strainInc)) :: par_E, par_nu, par_G, par_K

      lAnneal = jInfoArray(i_info_AnnealFlag) 
      iLayer = jInfoArray(i_info_layer)
      kspt   = jInfoArray(i_info_kspt)
      intPt  = jInfoArray(i_info_Intpt)
      iUpdateEffMod = jInfoArray(i_info_effModDefn)
      iElemNumStartLoc = jInfoArray(i_info_ElemNumStartLoc)
      ptrjElemNum = loc(jInfoArray(iElemNumStartLoc))

      if ( (ndir .ne. 3) .or. ((nshr .ne. 1) .and. (nshr .ne. 3)) ) then
        call XPLB_ABQERR(-3,"Error: wrong number of stress components: %I + %I", [ndir, nshr], 0, 0)
      end if

      err = invoke_pt_vumat_batch(trim(cmname)//".pt"//c_null_char, &
               & defgradNew, nblock, ndir, nshr, props, nprops, enerInternNew, stressNew)

      return
end
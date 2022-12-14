/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*! \file Unpacker_registry.C
  \brief Register dsp::Unpacker-derived classes for use in this file
    
  Classes that inherit dsp::Unpacker may be registered for use by
  utilizing the Registry::List<dsp::Unpacker>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


/*! built-in FloatUnpacker reads the format output by Dump operation */
#include "dsp/FloatUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FloatUnpacker> dump;


#if HAVE_apsr

#include "dsp/APSRTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::APSRTwoBitCorrection> apsr2;

#include "dsp/APSRFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::APSRFourBit> apsr4;

#include "dsp/APSREightBit.h"
static dsp::Unpacker::Register::Enter<dsp::APSREightBit> apsr8;

#endif

#if HAVE_asp
#include "dsp/ASPUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::ASPUnpacker> asp;
#endif

#if HAVE_bcpm
#include "dsp/BCPMUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BCPMUnpacker> registry_bcpm;
#endif

#if HAVE_boreas
#include "dsp/BoreasSurveyUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BoreasSurveyUnpacker> boreas_survey;
#include "dsp/BoreasVoltageUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BoreasVoltageUnpacker> boreas_voltage;
#endif

#if HAVE_bpsr
#include "dsp/BPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BPSRUnpacker> bpsr;
#include "dsp/BPSRCrossUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::BPSRCrossUnpacker> bpsrcross;
#endif

#if HAVE_caspsr
#include "dsp/CASPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CASPSRUnpacker> caspsr;
#endif

#if HAVE_ska1
#include "dsp/AAVS2Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::AAVS2Unpacker> aavs2;
#include "dsp/CBFPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CBFPSRUnpacker> cbfpsr;
#include "dsp/SKA1Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::SKA1Unpacker> ska1;
#include "dsp/LFAASPEADUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::LFAASPEADUnpacker> lfaaspead;
#endif

#if HAVE_uwb
#include "dsp/UWBUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::UWBUnpacker> uwb;
#include "dsp/UWBFloatUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::UWBFloatUnpacker> uwbfloat;
#include "dsp/UWBFourBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::UWBFourBitUnpacker> uwbfourbit;
#include "dsp/UWBEightBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::UWBEightBitUnpacker> uwbeightbit;
#endif

#if HAVE_cpsr
#include "dsp/CPSRTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::CPSRTwoBitCorrection> cpsr;
#endif

#if HAVE_cpsr2
#include "dsp/CPSR2TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2TwoBitCorrection> cpsr2;
#endif

#if HAVE_dummy
#include "dsp/DummyUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::DummyUnpacker> dummy;
#include "dsp/DummyFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::DummyFourBit> dummy4;
#endif

#if HAVE_fadc
#include "dsp/FadcUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FadcUnpacker> fadc;
#include "dsp/FadcTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::FadcTwoBitCorrection> fadc2;
#endif

#if HAVE_gmrt
#include "dsp/GMRTUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTUnpacker> gmrt;
#include "dsp/GMRTFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTFourBit> gmrt4;
#include "dsp/GMRTFilterbank16.h"
static dsp::Unpacker::Register::Enter<dsp::GMRTFilterbank16> gmrt16;
#endif

#if HAVE_guppi
#include "dsp/GUPPIUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPIUnpacker> guppi;
#include "dsp/GUPPIFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPIFourBit> guppi4;
#include "dsp/GUPPITwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPITwoBitCorrection> guppi2;
#include "dsp/GUPPITwoBitCorrectionComplex.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPITwoBitCorrectionComplex> guppi2c;
#endif

#if HAVE_kat
#include "dsp/KAT7Unpacker.h"
#include "dsp/MeerKATUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::KAT7Unpacker> kat7;
static dsp::Unpacker::Register::Enter<dsp::MeerKATUnpacker> meerkat;
#endif

#if HAVE_lofar_dal
#include "dsp/LOFAR_DALUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::LOFAR_DALUnpacker> lofar_dal;
#endif

#if HAVE_lbadr
#include "dsp/SMROTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::SMROTwoBitCorrection> lbadr;
#endif

#if HAVE_lbadr64
#include "dsp/LBADR64_TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::LBADR64_TwoBitCorrection> lbadr64;
#endif

#if HAVE_lump
#include "dsp/LuMPUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::LuMPUnpacker> lump;
#endif

#if HAVE_lwa
#include "dsp/LWAUnpacker.h"
static Registry::List<dsp::Unpacker>::Enter<dsp::LWAUnpacker> lwa;
#endif

#if HAVE_spda1k
#include "dsp/spda1k_Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::SPDA1K_Unpacker> spda1k;
#endif

#if HAVE_mark4
#include "dsp/Mark4TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::Mark4TwoBitCorrection> mark4;
#endif

#if HAVE_mark5
#include "dsp/Mark5Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::Mark5Unpacker> mark5_general;
#include "dsp/Mark5TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::Mark5TwoBitCorrection> mark5;
#endif

#if HAVE_mark5b
#include "dsp/Mark5bUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::Mark5bUnpacker> mark5b;
#endif

#if HAVE_maxim
#include "dsp/MaximUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::MaximUnpacker> maxim;
#endif

#if HAVE_mini
#include "dsp/MiniUnpack.h"
static dsp::Unpacker::Register::Enter<dsp::MiniUnpack> miniunpack;
#endif

#if HAVE_mopsr
#include "dsp/MOPSRUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::MOPSRUnpacker> mopsr;
#endif

#if HAVE_mwa
#include "dsp/EDAFourBit.h"
static dsp::Unpacker::Register::Enter<dsp::EDAFourBit> eda4bit;
#endif

#if HAVE_pmdaq
#include "dsp/OneBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::OneBitCorrection>  pmdaq;
#endif

#if HAVE_puma
#include "dsp/PuMaTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::PuMaTwoBitCorrection>  puma;
#endif

#if HAVE_puma2
#include "dsp/PuMa2Unpacker.h"
static dsp::Unpacker::Register::Enter<dsp::PuMa2Unpacker> puma2;
#endif

#if HAVE_s2
#include "dsp/S2TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::S2TwoBitCorrection>  s2;
#endif

#if HAVE_sigproc
#include "dsp/SigProcUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::SigProcUnpacker> sigproc;
#endif

#if HAVE_spigot
#include "dsp/ACFUnpack.h"
static dsp::Unpacker::Register::Enter<dsp::ACFUnpack> spigot;
#endif

#if HAVE_fits
#include "dsp/GUPPIFITSUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GUPPIFITSUnpacker> guppifits;
#include "dsp/FITSUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::FITSUnpacker> fits;
#endif


#if HAVE_emerlin
#include "dsp/EmerlinUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::EmerlinUnpacker> emerlin;
#endif

#if HAVE_vdif
#include "dsp/VDIFTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFTwoBitCorrection> vdif;
#include "dsp/VDIFTwoBitCorrectionMulti.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFTwoBitCorrectionMulti> vdif_multi;
#include "dsp/VDIFFourBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFFourBitUnpacker> vdif4;
#include "dsp/VDIFEightBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFEightBitUnpacker> vdif8;
#include "dsp/VDIFnByteUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::VDIFEightBitUnpacker> vdifN;
#endif

#if HAVE_wapp
#include "dsp/WAPPUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::WAPPUnpacker> wapp;
#endif

// ///////////////////////////////////////////////////////////////////////////
//
// The rest of these need work or seem to have disappeared
//
// ///////////////////////////////////////////////////////////////////////////

#if HAVE_CPSR2_4bit
#include "dsp/CPSR2FourBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2FourBitUnpacker> cpsr2_4;
#endif

#if HAVE_CPSR2_8bit
#include "dsp/CPSR2_8bitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::CPSR2_8bitUnpacker> cpsr2_8;
#endif

#if HAVE_vsib
#include "dsp/VSIBTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::VSIBTwoBitCorrection>  vsib;
#endif

#if HAVE_DUMBLBA
#include "dsp/DumbLBAUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::Dumb_LBAUnpacker> unpacker_register_dumblba;
#endif

#if HAVE_k5
#include "dsp/K5TwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::K5TwoBitCorrection>  k5;
#endif

/*
  Generic eight-bit unpacker is used if no other eight-bit unpacker steps up
*/

#include "dsp/GenericEightBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GenericEightBitUnpacker> gen8bit;

/*
  Generic four-bit unpacker is used if no other four-bit unpacker steps up
*/

#include "dsp/GenericFourBitUnpacker.h"
static dsp::Unpacker::Register::Enter<dsp::GenericFourBitUnpacker> gen4bit;

/*
  Generic two-bit unpacker is used if no other two-bit unpacker steps up
*/

#include "dsp/GenericTwoBitCorrection.h"
static dsp::Unpacker::Register::Enter<dsp::GenericTwoBitCorrection> gen2bit;


/*
  get_registry is defined here to ensure that this file is linked
*/
dsp::Unpacker::Register& dsp::Unpacker::get_register()
{
  return Register::get_registry();
}


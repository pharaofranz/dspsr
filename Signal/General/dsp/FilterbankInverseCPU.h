//-*-C++-*-
#pragma once

#ifndef __FilterbankInverseCPU_hpp
#define __FilterbankInverseCPU_hpp

#include "dsp/FilterbankEngine.h"
#include "dsp/Scratch.h"
//
#include "dsp/Transformation.h"

class FilterbankInverseEngineCPU : public dsp::Filterbank::Engine
{
	public:
		FilterbankInverseEngineCPU() {};
		~FilterbankInverseEngineCPU() {};

		void setup(dsp::Filterbank* filterbank);
		void setup_primary(dsp::Filterbank* filterbank);

		void set_scratch (float* scratch);

		void perform_primary(const dsp::TimeSeries* in, dsp::TimeSeries* out,
				uint64_t npart, uint64_t in_step, uint64_t out_step);
		void perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,
				uint64_t npart, uint64_t in_step, uint64_t out_step);

		void finish();

	private:
		unsigned _frequencyResolution;
		unsigned _nChannelSubbands;
		bool _isRealToComplex;
		unsigned _nPointsToKeep;
		double _scaleFactor;
		unsigned _nFftPoints;
		unsigned _nFftSamples;
		unsigned _nSampleOverlap;
		unsigned _nSampleStep;
	 	unsigned _nFilterPosition;
		unsigned _nInputChannels;
		unsigned _nPolarization;
		//
		FTransform::Plan* _forward;
		FTransform::Plan* _backward;
		const dsp::Response* _response;
		//
		//float* _scratch;
		float* _complexSpectrum[2];
		float* _complexTime;
		float* _windowedTimeDomain;
		dsp::Scratch* _scratchSpace;
};

#endif

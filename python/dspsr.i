%module dspsr
%feature("flatnested", "1");

%{
#define SWIG_FILE_WITH_INIT
#include <iostream>
#include "numpy/noprefix.h"

#include "dsp/Operation.h"
#include "dsp/File.h"
#include "Reference.h"
#include "dsp/Transformation.h"
#include "dsp/Unpacker.h"
#include "dsp/Observation.h"
#include "dsp/DataSeries.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"

#include "dsp/BitSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"

#include "dsp/Shape.h"
#include "dsp/Response.h"
#include "dsp/Dedispersion.h"

#include "dsp/Convolution.h"
#include "dsp/FilterbankConfig.h"
#include "dsp/Filterbank.h"
#include "dsp/InverseFilterbankConfig.h"
#include "dsp/InverseFilterbank.h"

#include "dsp/SingleThread.h"
#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"

using namespace dsp;

%}

// Language independent exception handler
%include exception.i

%include std_string.i
%include stdint.i

using namespace std;

%exception {
    try {
        $action
    } catch(Error& error) {
        // Deal with out-of-range errors
        if (error.get_code()==InvalidRange)
            SWIG_exception(SWIG_IndexError, error.get_message().c_str());
        else
            SWIG_exception(SWIG_RuntimeError,error.get_message().c_str());
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError,"Unknown exception");
    }
}

%init %{
  import_array();
%}


// Declare functions that return a newly created object
// (Helps memory management)
//%newobject Pulsar::Archive::new_Archive;
//%newobject Pulsar::Archive::load;

// Track any pointers handed off to python with a global list
// of Reference::To objects.  Prevents the C++ routines from
// prematurely destroying objects by effectively making python
// variables act like Reference::To pointers.
%feature("ref")   Reference::Able "pointer_tracker_add($this);"
%feature("unref") Reference::Able "pointer_tracker_remove($this);"

%header %{
std::vector< Reference::To<Reference::Able> > _pointer_tracker;
void pointer_tracker_add(Reference::Able *ptr) {
    _pointer_tracker.push_back(ptr);
}
void pointer_tracker_remove(Reference::Able *ptr) {
    std::vector< Reference::To<Reference::Able> >::iterator it;
    for (it=_pointer_tracker.begin(); it<_pointer_tracker.end(); it++)
        if ((*it).ptr() == ptr) {
            _pointer_tracker.erase(it);
            break;
        }
}
%}

// Non-wrapped stuff to ignore
%ignore dsp::IOManager::add_extensions(Extensions*);
%ignore dsp::IOManager::combine(const Operation*);
%ignore dsp::IOManager::set_scratch(Scratch*);
%ignore dsp::BitSeries::set_memory(Memory*);
%ignore dsp::Detection::set_engine(Engine*);
%ignore dsp::Convolution::set_engine(Engine*);
%ignore dsp::Observation::verbose_nbytes(uint64_t) const;
%ignore dsp::Observation::set_deripple(const std::vector<dsp::FIRFilter>&);
%ignore dsp::Observation::get_deripple();
// dsp::SingleThread::cerr breaks SWIG
%ignore dsp::SingleThread::cerr;

%rename (TimeSeries_Engine) dsp::TimeSeries::Engine;
%rename (Detection_Engine) dsp::Detection::Engine;
%rename (Convolution_Engine) dsp::Convolution::Engine;
%rename (Filterbank_Engine) dsp::Filterbank::Engine;
%rename (Filterbank_Config) dsp::Filterbank::Config;
%rename (InverseFilterbank_Engine) dsp::InverseFilterbank::Engine;
%rename (InverseFilterbank_Config) dsp::InverseFilterbank::Config;
%rename (SingleThread_Config) dsp::SingleThread::Config;
%rename (LoadToFold_Config) dsp::LoadToFold::Config;
// Return psrchive's Estimate class as a Python tuple
%typemap(out) Estimate<double> {
    PyTupleObject *res = (PyTupleObject *)PyTuple_New(2);
    PyTuple_SetItem((PyObject *)res, 0, PyFloat_FromDouble($1.get_value()));
    PyTuple_SetItem((PyObject *)res, 1, PyFloat_FromDouble($1.get_variance()));
    $result = (PyObject *)res;
}
%typemap(out) Estimate<float> = Estimate<double>;

// Return psrchive's MJD class as a Python double.
// NOTE this loses precision so may not be appropriate for all cases.
%typemap(out) MJD {
    $result = PyFloat_FromDouble($1.in_days());
}

// Convert various enums to/from string
%define %map_enum(TYPE)
%typemap(out) Signal:: ## TYPE {
    $result = PyString_FromString( TYPE ## 2string($1).c_str());
}
%typemap(in) Signal:: ## TYPE {
    try {
        $1 = Signal::string2 ## TYPE (PyString_AsString($input));
    } catch (Error &error) {
        SWIG_exception(SWIG_RuntimeError,error.get_message().c_str());
    }
}
%enddef


// the `arg2` name may not be persistent in different SWIG versions, or
// may be modified if the source code is modified
// %pythonprepend dsp::LoadToFold::set_configuration %{
//     if hasattr(arg2, "thisown"):
//         arg2.thisown = 0
// %}
//
// %pythonprepend dsp::LoadToFold::LoadToFold %{
//     if hasattr(config, "thisown"):
//         config.thisown = 0
// %}

%map_enum(State)
%map_enum(Basis)
%map_enum(Scale)
%map_enum(Source)
%map_enum(Behaviour)
/* %rename("%(title)s", %$isenumitem) ""; */



// Header files included here will be wrapped
%include "dsp/Operation.h"
%include "ReferenceAble.h"
%include "dsp/Transformation.h"

%template(TransformationTimeSeriesTimeSeries) dsp::Transformation<dsp::TimeSeries, dsp::TimeSeries>;
/* template class dsp::Transformation<dsp::TimeSeries, dsp::TimeSeries>; */

%include "dsp/Observation.h"
%include "dsp/Input.h"
%include "dsp/Seekable.h"
%include "dsp/File.h"
%include "dsp/DataSeries.h"
%include "dsp/IOManager.h"
%include "dsp/Input.h"
%include "dsp/BitSeries.h"
%include "dsp/TimeSeries.h"
%include "dsp/Detection.h"
%include "dsp/Shape.h"
%include "dsp/Response.h"
%include "dsp/Dedispersion.h"

%include "dsp/Convolution.h"
%include "dsp/Filterbank.h"
%include "dsp/FilterbankConfig.h"
%include "dsp/InverseFilterbank.h"
%include "dsp/InverseFilterbankConfig.h"

%include "dsp/SingleThread.h"
%include "dsp/LoadToFold1.h"
%include "dsp/LoadToFoldConfig.h"

// Python-specific extensions to the classes:
%extend dsp::TimeSeries
{
    // Return a numpy array view of the data.
    // This points to the data array, not a separate copy.
    PyObject *get_dat(unsigned ichan, unsigned ipol)
    {
        PyArrayObject *arr;
        float *ptr;
        npy_intp dims[2];

        dims[0] = self->get_ndat();
        dims[1] = self->get_ndim();
        ptr = self->get_datptr(ichan, ipol);
        arr = (PyArrayObject *)                                         \
            PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT, (char *)ptr);
        if (arr == NULL) return NULL;
        return (PyObject *)arr;
    }

    // Get the frac MJD part of the start time
    double get_start_time_frac()
    {
        return self->get_start_time().fracday();
    }
}
/* %rename(Transformation_outofplace) outofplace; */

/* %{
typedef dsp::Convolution::Engine Engine;
%} */

#ifndef __ScalarAdd_h
#define __ScalarAdd_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  class ScalarAdd : public Transformation<TimeSeries, TimeSeries> {

  public:

    ScalarAdd ();

    ~ScalarAdd ();

    void prepare ();

    void set_scalar (float _scalar) { scalar = _scalar; }

    float get_scalar () const { return scalar; }

  protected:

    virtual void transformation ();

  private:

    float scalar;

  };
}
#endif

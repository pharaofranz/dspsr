//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/PlasmaResponseProduct.h

#ifndef __PlasmaResponseProduct_h
#define __PlasmaResponseProduct_h

#include "dsp/PlasmaResponse.h"

namespace dsp {

  //! Represents a product of PlasmaResponse instances
  /*! 
    The dimensions of the product will contain the dimensions of
    each term in the product, as defined by:

   - the total required frequency resolution per output frequency channel
   - the largest dimension: matrix > dual polarization > single; complex > real

  */
  class PlasmaResponseProduct: public PlasmaResponse
  {

  public:

    //! Default constructor
    PlasmaResponseProduct ();

    //! Destructor
    ~PlasmaResponseProduct ();

    //! Prepare a product to match the input
    void prepare (const Observation* input, unsigned nchan);

    //! Add a response to the product
    void add_response (PlasmaResponse* response);

    //! Return the effective delay for the given frequency
    double delay_time (double freq) const;
    
  protected:

    //! The responses
    std::vector< Reference::To<PlasmaResponse> > response;

    //! Flag set true when a component has changed
    bool component_changed;

    //! Called when a component has changed
    void set_component_changed (const Response& response);

    //! Construct by combining the reponses of the components
    void build (unsigned ndat, unsigned nchan);
    
  };

}

#endif

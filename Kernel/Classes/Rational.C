/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Rational.h"

std::istream& operator >> (std::istream& in, Rational& r)
{
  char divide = 0;
  in >> r.numerator >> divide >> r.denominator;
  if (divide != '/')
    in.setstate(std::istream::failbit);

  return in;
}

std::ostream& operator << (std::ostream& out, const Rational& r)
{
  out << r.numerator << "/" << r.denominator;
  return out;
}


Rational::Rational (int num, int den)
{
  numerator = num;
  denominator = den;
}

double Rational::doubleValue( ) const
{
  return double(numerator) / double(denominator);
}

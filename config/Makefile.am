
#############################################################################
#
# This Makefile smooths over various bugs and bumps and other features
# that break the other GNU autotools-generated Makefiles
#

EXTRA_DIST = backends.default formats.sh Makefile.aclocal cudalt.py

all: aclocal_update

libs:

MAKEFILE_ACLOCAL = $(top_srcdir)/config/Makefile.aclocal

include $(MAKEFILE_ACLOCAL)

$(MAKEFILE_ACLOCAL): $(PSRCHIVE_ACLOCAL)/Makefile.aclocal
	cp $(PSRCHIVE_ACLOCAL)/Makefile.aclocal $(MAKEFILE_ACLOCAL)

$(PSRCHIVE_ACLOCAL)/Makefile.aclocal:
	@echo; echo; echo; \
	echo A required configuration file is missing; \
	echo Please update your PSRCHIVE installation; \
	echo; echo; echo; \
	exit -1



#!/bin/gnuplot

alimikhail(t,theta) = log((1-theta*(1-t))/t)
clayton(t,theta) = 1./theta*(t**(-theta)-1)
frank(t,theta) = -log((exp(-theta*t)-1)/(exp(-theta)-1))
gumbel(t,theta) = (-log(t))**theta
independent(t,theta) = -log(t)
joe(t,theta) = -log(1-(1-t)**theta)

ialimikhail(t,theta) = (1-theta)/(exp(t)-theta)
iclayton(t,theta) = (1+theta*t)**(-1./theta)
ifrank(t,theta) = -1./theta*log(1+exp(-t)*(exp(-theta)-1))
igumbel(t,theta) = exp(-t**(1./theta))
iindependent(t,theta) = exp(-t)
ijoe(t,theta) = 1-(1-exp(-t))**(1./theta)


options_alimikhail="-1.0 -0.5 0.0 0.5 0.9"
options_clayton="-1.0 -0.5 0.5 1.0 2.0"
options_frank="-2.0 -0.5 0.5 2.0 4.0"
options_gumbel="1.0 2.0 3.0 4.0 5.0"
options_independent="0.0"
options_joe="1.0 2.0 4.0 8.0 16.0"

set multiplot layout 2, 6 title "Copula" font ",14"
set xrange [0:1]

set title "Ali-Mikhail-Jak"
options = options_alimikhail
plot for [i=1:words(options)] alimikhail(x,word(options_alimikhail,i)) ti sprintf("theta=%s",word(options,i))

set title "Clayton"
options = options_clayton
set yrange [0:50]
plot for [i=1:words(options)] clayton(x,word(options,i)) ti sprintf("theta=%s",word(options,i))
unset yrange

set title "Frank"
options = options_frank
plot for [i=1:words(options)] frank(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Gumbel"
options = options_gumbel
plot for [i=1:words(options)] gumbel(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Independence"
options = options_independent
plot for [i=1:words(options)] independent(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Joe"
options = options_joe
plot for [i=1:words(options)] joe(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

#unset multiplot


#options="0.5 1.0 2.0 4.0"
#set multiplot layout 2, 3 title "Copula" font ",14"
set xrange [0:100]

set title "Ali-Mikhail-Jak"
options = options_alimikhail
plot for [i=1:words(options)] ialimikhail(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Clayton"
options = options_clayton
plot for [i=1:words(options)] iclayton(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Frank"
options = options_frank
plot for [i=1:words(options)] ifrank(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Gumbel"
options = options_gumbel
plot for [i=1:words(options)] igumbel(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Independence"
options = options_independent
plot for [i=1:words(options)] iindependent(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

set title "Joe"
options = options_joe
plot for [i=1:words(options)] ijoe(x,word(options,i)) ti sprintf("theta=%s",word(options,i))

unset multiplot

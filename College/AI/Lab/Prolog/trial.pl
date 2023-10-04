parent(pam, bob).
parent(bob, ann).
parent(bob, pat).
parent(tom,maty).
parent(tom, bob).
parent(tom, lizz).
parent(pat,jim).

female(pam).
female(pat).
female(lizz).
female(ann).
female(maty).

male(tom).
male(bob).
male(jim).

grandparent(X,Y) :- 
    parent(X,Z), 
    parent(Z,Y).

mother(X,Y) :- 
    parent(X,Y), 
    female(X).

father(X,Y) :- 
    parent(X,Y), 
    male(Y).

siblings(X,Y) :- 
    parent(Z,X), 
    parent(Z,Y), 
    X \= Y.

sister(X,Y) :- 
    siblings(X,Y), 
    female(X).

aunt(X,Y) :- 
    parent(Z,Y), 
    sister(X,Z).

uncle(X,Y) :- 
    parent(Z,Y), 
    brother(X,Z).

grandmother(X,Y) :- 
    grandparent(X,Y), 
    female(X).

grandfather(X,Y) :- 
    grandparent(X,Y), 
    male(X).

daughter(X,Y) :- 
    parent(Y,X), 
    female(X).

son(X,Y) :- 
    parent(Y,X), 
    male(X).

child(X,Y) :- 
    parent(Y,X).

predecessor(X,Y) :- 
    parent(X,Y).

predecessor(X,Y) :- 
    parent(X,Z), 
    predecessor(Z,Y).
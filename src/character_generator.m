function [characters] = character_generator()
c = [ 
    0 0 0 0 0 ...
    0 0 0 0 0 ...
    0 1 1 1 1 ...
    1 0 0 0 0 ...
    1 0 0 0 0 ...
    1 0 0 0 0 ...
    0 1 1 1 1 ]';
e = [
    0 0 0 0 0 ...
    0 0 0 0 0 ...
    0 1 1 1 0 ...
    1 0 0 0 1 ...
    1 1 1 1 0 ...
    1 0 0 0 0 ...
    0 1 1 1 1 ]';
d = [
    0 0 0 0 1 ...
    0 0 0 0 1 ...
    0 1 1 1 1 ...
    1 0 0 0 1 ...
    1 0 0 0 1 ...
    1 0 0 0 1 ...
    0 1 1 1 1 ]';
r = [ 
    0 0 0 0 0 ...
    0 0 0 0 0 ...
    0 1 1 1 1 ...
    1 0 0 0 0 ...
    1 0 0 0 0 ...
    1 0 0 0 0 ...
    1 0 0 0 0 ]';
i = [
    0 0 1 0 0 ...
    0 0 0 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ]';
l = [
    0 1 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 0 0 ...
    0 0 1 1 0 ]';
o = [
    0 0 0 0 0 ...
    0 0 0 0 0 ...
    0 1 1 1 0 ...
    1 0 0 0 1 ...
    1 0 0 0 1 ...
    1 0 0 0 1 ...
    0 1 1 1 0 ]';   
characters = [c, e, d, r, i, c, l, o, o, d, prprob];
===
- reuse code between shaders rather than copying it
  
- debug menu/text

- join post_effects.wgsl with bloom.wgsl since they are both post-effects

- make world persistent - LocalStorage on web and files in `chunks` dir in native (load on startup and when creating chunks that arent loaded yet)

- do we even need more than 1 shader if 1 shader file can have multiple vertex/fragment input functions?

- use replacable constants ins hadows (pass values for constants from odin)

!!!
- remove align 16 ( ithink it affects struct pos in aray and that it)
!!!
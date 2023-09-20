{
  description = "Cartpole in Julia";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      packages.default = pkgs.writeScriptBin "run_pluto" ''
        ${pkgs.julia}/bin/julia -E "import Pkg; Pkg.add(\"Pluto\"); using Pluto; Pluto.run(notebook=\"rl_notebook.jl\")"
      '';
      formatter = pkgs.alejandra;
    });
}

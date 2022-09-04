using ACE1x
using Documenter

DocMeta.setdocmeta!(ACE1x, :DocTestSetup, :(using ACE1x); recursive=true)

makedocs(;
    modules=[ACE1x],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACE1x.jl/blob/{commit}{path}#{line}",
    sitename="ACE1x.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACE1x.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACE1x.jl",
    devbranch="main",
)
